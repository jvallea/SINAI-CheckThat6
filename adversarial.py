# Ejecutar en el terminal
#
# git clone https://github.com/piotrmp/BODEGA
# git clone https://gitlab.clarin-pl.eu/syntactic-tools/lambo.git
# pip install OpenAttack editdistance bert-score git+https://github.com/lucadiliello/bleurt-pytorch.git ./lambo
# 
# mkdir ./content
# mv ./BODEGA/ ./content/
# mkdir ./content/clef2024-checkthat-lab
# cd ./content/clef2024-checkthat-lab
# 
# git init
# git remote add -f origin https://gitlab.com/checkthat_lab/clef2024-checkthat-lab.git
# git sparse-checkout init
# git sparse-checkout set "task6/incrediblAE_public_release"
# git pull origin main
# 
# cd ../../
# mv ./content/clef2024-checkthat-lab/task6/incrediblAE_public_release ./content/BODEGA/incrediblAE_public_release
# mkdir ./content/BODEGA/outputs
# 
# pip install homoglyphs shap keybert nltk spacy
# python -m spacy download en_core_web_lg

# NOTA: Este archivo debe estar en ./content/BODEGA/

#############################################################################################################
#############################################################################################################
#############################################################################################################

###############
##  ARGPARSE ##
###############
import argparse

parser = argparse.ArgumentParser(description='BODEGA Adversarial Attack')
parser.add_argument('--t', type=str, default='PR2', help='Task name (PR2, HN, FC, RD, C19)')
parser.add_argument('--v', type=str, default='BERT', help='Victim model name (BERT, BiLSTM, surprise)')
parser.add_argument('--s', type=str, default='memory_bf', help='Selection Algorithm type (shap, shap_hybrid, bf, memory_bf, keybert, keybert_hybrid)')
parser.add_argument('--h', type=str, default="homo", help="Attack type (homo, syn, inv, mix)")

args = parser.parse_args()

print(args.t)
print(args.v)
print(args.s)
print(args.h)

###############
### IMPORTS ###
###############

import gc
import os
import pathlib
import sys
import time
import random
import numpy as np

import OpenAttack
import torch
import datasets
from datasets import Dataset

from OpenAttack.tags import Tag
from OpenAttack.text_process.tokenizer import PunctTokenizer

from metrics.BODEGAScore import BODEGAScore
from utils.data_mappings import dataset_mapping, dataset_mapping_pairs, SEPARATOR_CHAR
from utils.no_ssl_verify import no_ssl_verify
from victims.bert import VictimBERT
from victims.bert import readfromfile_generator as BERT_readfromfile_generator
from victims.bilstm import VictimBiLSTM
from victims.caching import VictimCache
from victims.unk_fix_wrapper import UNK_TEXT
from keybert import KeyBERT

#imports for BodegaAttackEval wrapper
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
from tqdm import tqdm
from OpenAttack.utils import visualizer, result_visualizer, get_language, language_by_name
from OpenAttack.tags import *

#############################################################################################################
#############################################################################################################
#############################################################################################################


###############################
### Roberta Base Classifier ###
###############################

import numpy

from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoConfig
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.data_mappings import SEPARATOR
import pathlib

BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 5
MAX_BATCHES = -1
pretrained_model = "roberta-base"

def trim(text, tokenizer):
    offsets = tokenizer(text, truncation=True, max_length=MAX_LEN + 10, return_offsets_mapping=True)['offset_mapping']
    limit = len(text)
    if len(offsets) > MAX_LEN:
        limit = offsets[512][1]
    return text[:limit]


def roberta_readfromfile_generator(subset, dir, with_pairs=False, trim_text=False):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    for line in open(dir / (subset + '.tsv')):
        parts = line.split('\t')
        label = int(parts[0])
        if not with_pairs:
            text = parts[2].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            if trim_text:
                text = trim(text, tokenizer)
            yield {'fake': label, 'text': text}
        else:
            text1 = parts[2].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            text2 = parts[3].strip().replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
            if trim_text:
                text1 = trim(text1, tokenizer)
                text2 = trim(text2, tokenizer)
            yield {'fake': label, 'text1': text1, 'text2': text2}


def eval_loop(model, eval_dataloader, device, skip_visual=False):
    print("Evaluating...")
    model.eval()
    progress_bar = tqdm(range(len(eval_dataloader)), ascii=True, disable=skip_visual)
    correct = 0
    size = 0
    TPs = 0
    FPs = 0
    FNs = 0
    for i, batch in enumerate(eval_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        # print(logits)
        # a = input()
        pred = torch.argmax(logits, dim=-1).detach().to(torch.device('cpu')).numpy()
        Y = batch["labels"].to(torch.device('cpu')).numpy()
        eq = numpy.equal(Y, pred)
        size += len(eq)
        correct += sum(eq)
        TPs += sum(numpy.logical_and(numpy.equal(Y, 1.0), numpy.equal(pred, 1.0)))
        FPs += sum(numpy.logical_and(numpy.equal(Y, 0.0), numpy.equal(pred, 1.0)))
        FNs += sum(numpy.logical_and(numpy.equal(Y, 1.0), numpy.equal(pred, 0.0)))
        progress_bar.update(1)

        # print(Y)
        # print(pred)
        # a = input()

        if i == MAX_BATCHES:
            break
    print('Accuracy: ' + str(correct / size))
    print('F1: ' + str(2 * TPs / (2 * TPs + FPs + FNs)))
    print(correct, size, TPs, FPs, FNs)

    results = {
        'Accuracy': correct/size,
        'F1': 2 * TPs / (2 * TPs + FPs + FNs)
    }
    return results


class VictimRoBERTa(OpenAttack.Classifier):
    def __init__(self, path, task, device=torch.device('cpu')):
        self.device = device
        config = AutoConfig.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_config(config)
        self.model.load_state_dict(torch.load(path))
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.with_pairs = (task == 'FC' or task == 'C19')

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        try:
            probs = None
            # print(len(input_), input_)

            batched = [input_[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i in
                       range((len(input_) + BATCH_SIZE - 1) // BATCH_SIZE)]
            for batched_input in batched:
                if not self.with_pairs:
                    tokenised = self.tokenizer(batched_input, truncation=True, padding=True, max_length=MAX_LEN,
                                            return_tensors="pt")
                else:
                    parts = [x.split(SEPARATOR) for x in batched_input]
                    tokenised = self.tokenizer([x[0] for x in parts], [(x[1] if len(x) == 2 else '') for x in parts],
                                            truncation=True, padding=True,
                                            max_length=MAX_LEN,
                                            return_tensors="pt")
                with torch.no_grad():
                    tokenised = {k: v.to(self.device) for k, v in tokenised.items()}
                    outputs = self.model(**tokenised)
                probs_here = torch.nn.functional.softmax(outputs.logits, dim=-1).to(torch.device('cpu')).numpy()
                if probs is not None:
                    probs = numpy.concatenate((probs, probs_here))
                else:
                    probs = probs_here
            return probs
        except Exception as e:
            # Used for debugging
            raise

#############################################################################################################
#############################################################################################################
#############################################################################################################

using_mounted_drive = False
print('Cuda device available', torch.cuda.is_available())

#######################
### Submission File ###
#######################

class BodegaAttackEval(OpenAttack.AttackEval):
    '''
    wrapper for OpenAttack.AttackEval to produce a submission.tsv file for shared task evaluation

    To perform evaluation, we use a new method: eval_and_save_tsv() rather than the usual AttackEval.eval()
    submission.tsv file consists of 4 columns for each sample in attack set: succeeded, num_queries, original_text and modified text (newlines are escaped)

    '''
    def eval_and_save_tsv(self, dataset: Iterable[Dict[str, Any]], total_len : Optional[int] = None, visualize : bool = False, progress_bar : bool = False, num_workers : int = 0, chunk_size : Optional[int] = None, tsv_file_path: Optional[os.PathLike] = None):
        """
        Evaluation function of `AttackEval`.

        Args:
            dataset: An iterable dataset.
            total_len: Total length of dataset (will be used if dataset doesn't has a `__len__` attribute).
            visualize: Display a pretty result for each data in the dataset.
            progress_bar: Display a progress bar if `True`.
            num_workers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.

            tsv_file_path: path to save submission tsv

        Returns:
            A dict of attack evaluation summaries.

        """


        if hasattr(dataset, "__len__"):
            total_len = len(dataset)

        def tqdm_writer(x):
            return tqdm.write(x, end="")

        if progress_bar:
            result_iterator = tqdm(self.ieval(dataset, num_workers, chunk_size), total=total_len)
        else:
            result_iterator = self.ieval(dataset, num_workers, chunk_size)

        total_result = {}
        total_result_cnt = {}
        total_inst = 0
        success_inst = 0

        #list for tsv
        x_orig_list = []
        x_adv_list = []
        num_queries_list = []
        succeed_list = []

        # Begin for
        for i, res in enumerate(result_iterator):
            total_inst += 1
            success_inst += int(res["success"])

            if TAG_Classification in self.victim.TAGS:
                x_orig = res["data"]["x"]
                if res["success"]:
                    x_adv = res["result"]
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                        y_adv = probs[1]
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig, x_adv])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                        y_adv = int(preds[1])
                    else:
                        raise RuntimeError("Invalid victim model")
                else:
                    y_adv = None
                    x_adv = None
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([x_orig])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                    else:
                        raise RuntimeError("Invalid victim model")
                info = res["metrics"]
                info["Succeed"] = res["success"]
                if visualize:
                    if progress_bar:
                        visualizer(i + 1, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer, self.tokenizer)
                    else:
                        visualizer(i + 1, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write, self.tokenizer)

                #list for tsv
                succeed_list.append(res["success"])
                num_queries_list.append(res["metrics"]["Victim Model Queries"])
                x_orig_list.append(x_orig)

                if res["success"]:
                    x_adv_list.append(x_adv)
                else:
                    x_adv_list.append("ATTACK_UNSUCCESSFUL")



            for kw, val in res["metrics"].items():
                if val is None:
                    continue

                if kw not in total_result_cnt:
                    total_result_cnt[kw] = 0
                    total_result[kw] = 0
                total_result_cnt[kw] += 1
                total_result[kw] += float(val)
        # End for

        summary = {}
        summary["Total Attacked Instances"] = total_inst
        summary["Successful Instances"] = success_inst
        summary["Attack Success Rate"] = success_inst / total_inst
        for kw in total_result_cnt.keys():
            if kw in ["Succeed"]:
                continue
            if kw in ["Query Exceeded"]:
                summary["Total " + kw] = total_result[kw]
            else:
                summary["Avg. " + kw] = total_result[kw] / total_result_cnt[kw]

        if visualize:
            result_visualizer(summary, sys.stdout.write)


        #saving tsv
        if tsv_file_path is not None:
            with open(tsv_file_path, 'w') as f:
                f.write('succeeded' + '\t' + 'num_queries' + '\t' + 'original_text' + '\t' + 'modified_text' + '\t'+ '\n') #header
                for success, num_queries, x_orig, x_adv in zip(succeed_list, num_queries_list, x_orig_list, x_adv_list):
                    escaped_x_orig = x_orig.replace('\n', '\\n') #escaping newlines
                escaped_x_adv = x_adv.replace('\n', '\\n')
                f.write(str(success) + '\t' + str(num_queries) + '\t' + escaped_x_orig + '\t' + escaped_x_adv + '\t'+ '\n')

        return summary

#############################################################################################################
#############################################################################################################
#############################################################################################################

import homoglyphs as hg
import random
import copy
from datetime import datetime

import shap
from transformers import pipeline
import nltk
nltk.download('wordnet')
from nltk.wsd import lesk
from nltk.corpus import wordnet
import spacy
nlp = spacy.load("en_core_web_lg")
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

#####################
### Custom Attack ###
#####################
'''
This example code shows how to design a customized attack model (that shuffles the tokens in the original sentence).
Taken from https://github.com/thunlp/OpenAttack/blob/master/examples/custom_attacker.py
'''
class MyAttacker(OpenAttack.attackers.ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.lang_tag, Tag("get_pred", "victim") }

    def __init__(self, tokenizer = None):
        if tokenizer is None:
            tokenizer = PunctTokenizer()
        self.tokenizer = tokenizer
        self.lang_tag = OpenAttack.utils.get_language([self.tokenizer])

        # CONFIG        
        self.SEARCH_ALG = args.s
        self.HEURISTIC = args.h
        self.BF_SEARCH = "False"

        # CONSTRAINTS
        self.REMOVE_SIMPLE_CHARS = True
        self.MAX_SPLIT = 100 # Max split when is BigData
        self.LEN_EXPRESSIONS = 1 # Expression length when using Keybert
        self.STOPWORDS = 'english' # Don't select stopwords when SearchAlg is Keybert
        self.HOMOGLYPHS = 1
        self.ATTACK_NUMBER_BY_RANGE = 4

        # OTHER CONFIG
        self.model_pipeline = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        self.explainer = shap.Explainer(self.model_pipeline)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def attack(self, victim, input_, goal):

        print(f"[{datetime.now()}] ---/// STARTING ATTACK ///---")

        x_new = input_
        attacks = 0
        flag_word = 0

        if self.SEARCH_ALG in [ "bf", "memory_bf", "shap_hybrid", "keybert_hybrid" ]:
            self.BF_SEARCH == "True"

        if args.t in ["HN", "RD"] and self.SEARCH_ALG == "memory_bf":
            self.SEARCH_ALG = "big_data"
        if args.t in ["HN", "RD"] and self.SEARCH_ALG == "bf":
            self.SEARCH_ALG = "big_data_reduced"

        while (True):
            if self.SEARCH_ALG == "big_data" or self.SEARCH_ALG == "big_data_reduced":
                sentences_list = self.big_dataset_division(x_new) #Se divide en partes de MAX_SPLIT
                for i_sent in range(len(sentences_list)): #Se ataca cada token secuencialmente de la frase. Si es "big_data", se aplica la memoria.
                    splitted_sentence = sentences_list[i_sent].split()
                    if attacks > 0:
                        mod_sentences_list = sentences_list
                    else:
                        mod_sentences_list = sentences_list.copy()

                    for j_word in range(len(splitted_sentence)):
                        print("Palabras modificadas", " [frase ", i_sent,"]: ", (j_word+1), "/", len(splitted_sentence))
                        if attacks > 0:
                            splitted_sentence = self.attack_sentence(splitted_sentence, j_word)
                            ret_splitted_sentence = splitted_sentence
                        else:
                            ret_splitted_sentence = self.attack_sentence(splitted_sentence, j_word)
                        ret_sentence = " ".join(ret_splitted_sentence)
                        mod_sentences_list[i_sent] = ret_sentence
                        final_sentence = " ".join(mod_sentences_list)
                        if self.goal_check(final_sentence, goal) is not None:
                            return final_sentence
                if self.SEARCH_ALG == "big_data": #Si es "big_data", se repite el ataque con memoria. Similar a "memory_bf" pero dividiendo las frases.
                    attacks += 1
                    if attacks > 1:
                        return None
                else: #Si se trata del ataque reducido, no se continúa.
                    return None
            elif self.BF_SEARCH == "True":
                print("Starting iterative search...")
                splitted_sentence = x_new.split()
                x_new = self.brute_force(splitted_sentence, goal)
                if x_new is not None:
                    return x_new

            if self.SEARCH_ALG in [ "shap", "shap_hybrid" ]:
                if attacks > 0:
                    print("Ataque con shap fallido")
                    return None
                attacks += 1
                shap_splitted_sent, n_important_words_pos = self.shap_process_word(input_ , self.explainer, self.ATTACK_NUMBER_BY_RANGE)
                for i in n_important_words_pos:
                    print("Shap Attack: ", shap_splitted_sent[i])
                    shap_splitted_sent = self.attack_sentence(shap_splitted_sent, i)
                    x_new = "".join(shap_splitted_sent)
                    if self.goal_check(x_new, goal) is not None:
                        return x_new
            elif self.SEARCH_ALG == "memory_bf":
                # Almacenamos la frase con el modificador fijo
                print("Palabras modificadas:", (flag_word+1), "/", len(splitted_sentence))
                ff_splitted_sentece = self.attack_sentence(splitted_sentence, flag_word)
                x_new = " ".join(ff_splitted_sentece)
                flag_word += 1
                if flag_word == len(splitted_sentence) - 1:
                    return None
            elif self.SEARCH_ALG in [ "keybert", "keybert_hybrid" ]:
                if attacks > 0:
                    print("Ataque con keybert fallido")
                    return None
                attacks += 1
                keywords = self.keybert(input_, self.ATTACK_NUMBER_BY_RANGE)
                x_new = self.keybert_attack(input_, keywords)
                if self.goal_check(x_new, goal) is not None:
                    return x_new

    def brute_force(self, splitted_sentence, goal):
        for i in range(len(splitted_sentence)):
            ret_sentence = self.attack_sentence(splitted_sentence, i)
            x_new = " ".join(ret_sentence)
            if self.goal_check(x_new, goal) is not None:
                return x_new
        return None

    def big_dataset_division(self, sentence):
        splitted_sentence = sentence.split()
        number_words = len(splitted_sentence)
        division = number_words / self.MAX_SPLIT
        int_division = int(division)
        list_of_splitted_sentences = []

        for i in range (int_division + 1):
            original_lenght = i * self.MAX_SPLIT
            max_lenght = (i + 1) * self.MAX_SPLIT
            if i == 0:
                list_of_splitted_sentences.append(splitted_sentence[:self.MAX_SPLIT])
            elif i == int_division:
                list_of_splitted_sentences.append(splitted_sentence[original_lenght:])
            else:
                list_of_splitted_sentences.append(splitted_sentence[original_lenght:max_lenght])

        list_of_sentences = list_of_splitted_sentences.copy()

        count = 0
        for split in list_of_splitted_sentences:
            list_of_sentences[count] = ' '.join(split)
            count += 1

        return list_of_sentences

    def attack_sentence(self, splitted_sentence, i):
        word = splitted_sentence[i]
        mod_word = copy.deepcopy(word)
        ret_sentence = splitted_sentence.copy()
        if word[-1] == " ": mod_word = mod_word.replace(" ", "")
        if self.REMOVE_SIMPLE_CHARS == True:
            mod_word = self.delete_simple_chars(mod_word)
        if mod_word != "":
            if self.HEURISTIC == "syn":
                mod_word, attacked = self.synonym_attack(splitted_sentence, mod_word, i)
                # if not attacked:
                #     mod_word = self.homoglyph_attack(mod_word)
            if self.HEURISTIC == "inv":
                mod_word = self.invisible_char_attack(mod_word)
            elif self.HEURISTIC == "homo":
                mod_word = self.homoglyph_attack(mod_word)
            elif self.HEURISTIC == "mix":
                mod_word = self.homoglyph_attack(mod_word)
                mod_word = self.invisible_char_attack(mod_word)
        if word[-1] == " ": mod_word += " "
        ret_sentence[i] = mod_word
        return ret_sentence

    def goal_check(self, x_new, goal):
        # print("CHECKING sentence: ", x_new)
        y_new = victim.get_pred([ x_new ])
        # Check for attack goal
        if goal.check(x_new, y_new):
            print("Success! -- ", x_new)
            return x_new
        else:
            return None

    ####################
    ##   Homoglyphs   ##
    ####################
    def homoglyph_attack(self, word):
        letters = list(word)
        rand_letters_pos = []
        for i in range(self.HOMOGLYPHS):
            letter_pos = random.randrange(0,len(letters))
            while letter_pos in rand_letters_pos:
                if len(letters) <= self.HOMOGLYPHS or len(rand_letters_pos) == len(letters):
                    break
                letter_pos = random.randrange(0,len(letters))
            rand_letters_pos.append(letter_pos)

            hg_list = hg.Homoglyphs().get_combinations(letters[letter_pos])
            try:
                rand_homoglyph = random.randrange(1, len(hg_list))
            except:
                continue
            letters[letter_pos] = hg_list[rand_homoglyph]
        mod_word = "".join(letters)
        return mod_word

    ####################
    ## Invisible Char ##
    ####################
    def invisible_char_attack(self, word):
        invisible_character = "ᅟ"
        mod_word = invisible_character + word
        return mod_word

    ####################
    ##     KeyBert    ##
    ####################
    def keybert(self, sentence, ATTACK_NUMBER_BY_RANGE):
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, self.LEN_EXPRESSIONS), stop_words=self.STOPWORDS, top_n=self.ATTACK_NUMBER_BY_RANGE)
        return keywords

    def keybert_attack(self, sentence, keywords):
        sentence = sentence.lower()
        for keyword in keywords:
            word = keyword[0]
            mod_word = word
            if word[-1] == " ": mod_word = mod_word.replace(" ", "")
            if self.REMOVE_SIMPLE_CHARS == True:
                mod_word = self.delete_simple_chars(mod_word)
            if mod_word != "":
                if self.HEURISTIC == "syn":
                    mod_word, attacked = self.synonym_attack(sentence.split(), mod_word, sentence.split().index(word))
                    # if not attacked:
                    #     mod_word = self.homoglyph_attack(mod_word)
                if self.HEURISTIC == "inv":
                    mod_word = self.invisible_char_attack(mod_word)
                elif self.HEURISTIC == "homo":
                    mod_word = self.homoglyph_attack(mod_word)
                elif self.HEURISTIC == "mix":
                    mod_word = self.homoglyph_attack(mod_word)
                    mod_word = self.invisible_char_attack(mod_word)
            if word[-1] == " ": mod_word += " "
            sentence = sentence.replace(keyword[0], mod_word)
        print("---SENTENCE: ", sentence)
        return sentence

    ####################
    ##    Synonyms    ##
    ####################
    def synonym_attack(self, splitted_sentence, word, word_pos):
        sentence = " ".join(splitted_sentence)
        doc = nlp(sentence)
        pos = ""
        for token in doc:
            if token.text == word:
                if token.pos_ == "ADJ":
                    pos = "a"
                elif token.pos_ == "VERB":
                    pos = "v"
                elif token.pos_ == "NOUN":
                    pos = "n"
                elif token.pos_ == "ADV":
                    pos = "r"
        if pos == "":
            return word, False
        sent = sentence.split()
        synset = lesk(sent, word, pos)
        lowest_syn = None
        lowest_score = None
        try:
            for synonim in synset.lemma_names():

                if synonim.lower() != word.lower():

                    score = self.get_similarity_by_embeddings(word, synonim)
                    if lowest_score == None or score > lowest_score:
                        lowest_syn = synonim
                        lowest_score = score
        except Exception as e:
            print(e)
            return word, False
        if lowest_syn == None:
            return word, False

        # syn = wordnet.synsets(str(lowest_syn))[0]
        # hyponyms = syn.hyponyms()
        # hyponyms = hyponyms[0].lemma_names()[0]
        # print("Hyponims:", hyponyms)

        # return str(hyponyms), True
        return lowest_syn, True

    def get_similarity_by_embeddings(self, word,syn):
        encoding = self.tokenizer.batch_encode_plus(
            [word],
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            sentence_embedding = outputs.last_hidden_state.mean(dim=1)


        syn_encoding = self.tokenizer.batch_encode_plus(
            [syn],
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        syn_input_ids = syn_encoding['input_ids']
        syn_attention_mask = syn_encoding['attention_mask']

        with torch.no_grad():
            syn_outputs = self.model(syn_input_ids, attention_mask=syn_attention_mask)
            syn_sentence_embedding = syn_outputs.last_hidden_state.mean(dim=1)

        similarity_score = cosine_similarity(sentence_embedding, syn_sentence_embedding)

        return similarity_score[0][0]

    ####################
    ##  Constraints   ##
    ####################
    def delete_simple_chars(self, word):
        mod_word_list = list(word).copy()
        if len(mod_word_list) <= 1:
            return ""
        return "".join(mod_word_list)

    ####################
    ##      Shap      ##
    ####################
    # Ret: Splitted sentence in List, word position within splitted sentence.
    def shap_process_word(self, sentence, explainer, ATTACK_NUMBER_BY_RANGE):
        print("----Scanning:", sentence)
        shap_values = explainer([sentence])
        label, values = self.get_shap_label_and_values(shap_values)

        index_important_words = sorted(range(len(values)), key = lambda sub: values[sub])
        print("---Important Words sorted: ", index_important_words)

        if abs(values[index_important_words[0]]) > abs(values[index_important_words[-1]]):
            n_important_words_pos = index_important_words[:ATTACK_NUMBER_BY_RANGE]
        else:
            n_important_words_pos = index_important_words[-ATTACK_NUMBER_BY_RANGE:]
            n_important_words_pos.reverse()
        print("---Selected Words: ", n_important_words_pos)

        splitted_sentence = shap_values[0, :, label].data.copy()

        for i in n_important_words_pos:
            print("----IMPORTANT WORD:", splitted_sentence[i])

        return splitted_sentence.tolist(), n_important_words_pos
    
    def calculate_shap_values(self, shap_values, label):
        values = []
        for i in shap_values[0, :, label].values:
            values.append(float(i))
        return values

    def get_shap_label_and_values(self, shap_values):
        positive_values = self.calculate_shap_values(shap_values, "POSITIVE")
        negative_values = self.calculate_shap_values(shap_values, "NEGATIVE")

        if positive_values > negative_values:
            return "POSITIVE", positive_values
        else:
            return "NEGATIVE", negative_values

#############################################################################################################
#############################################################################################################
#############################################################################################################

############################
### Attack Configuration ###
############################

# determinism
random.seed(10)
torch.manual_seed(10)
np.random.seed(0)

# Change these variables to what you want
# task = 'PR2' # PR2, HN, FC, RD, C19
task = args.t
# victim_model = 'BERT' # BERT or BiLSTM or surprise
victim_model = args.v
using_custom_attacker = True # change to False if you want to test out OpenAttack's pre-implemented attackers (e.g. BERTattack)
attack = 'custom' # if using custom attack, this name can be whatever you want. If using pre-implemented attack, set to name of attacker ('BERTattack')

# misc variables - no need to change
targeted = False # this shared task evaluates performance in an untargeted scenario
visualize_adv_examples = True # prints adversarial samples as they are generated, showing the difference between original
using_first_n_samples = False # used when you want to evaluate on a subset of the full eval set.
first_n_samples = 10

#############################################################################################################
#############################################################################################################
#############################################################################################################

####################
### Run Attacker ###
####################

data_path =  pathlib.Path(f"./content/BODEGA/incrediblAE_public_release/{task}")
model_path = pathlib.Path(f"./content/BODEGA/incrediblAE_public_release/{task}/{victim_model}-512.pth")
out_dir = pathlib.Path("./content/BODEGA/outputs")



RESULTS_FILE_NAME = 'results_' + task + '_' + str(targeted) + '_' + attack + '_' + victim_model + '.txt' #stores BODEGA metrics
SUBMISSION_FILE_NAME = 'submission_' + task + '_' + str(targeted) + '_' + attack + '_' + victim_model + '.tsv' #stores original and modified text, to be submitted to shared task organizers

results_path = out_dir / RESULTS_FILE_NAME if out_dir else None
submission_path = out_dir / SUBMISSION_FILE_NAME if out_dir else None

if out_dir:
    if (out_dir / RESULTS_FILE_NAME).exists():
        print(f"Existing results file found. This script will overwrite previous file: {str(results_path)}")
    if submission_path.exists():
        print(f"Existing submission file found. This script will overwrite previous file: {str(submission_path)}")


# Prepare task data
with_pairs = (task == 'FC' or task == 'C19')

# Choose device
print("Setting up the device...")

using_TF = (attack in ['TextFooler', 'BAE'])
if using_TF:
    # Disable GPU usage by TF to avoid memory conflicts
    import tensorflow as tf

    tf.config.set_visible_devices(devices=[], device_type='GPU')

if torch.cuda.is_available():
    print('using GPU')
    victim_device = torch.device("cuda")
    attacker_device = torch.device("cuda")
else:
    victim_device = torch.device("cpu")
    attacker_device = torch.device('cpu')

# Prepare victim
print("Loading up victim model...")
if victim_model == 'BERT':
    victim = VictimCache(model_path, VictimBERT(model_path, task, victim_device))
    readfromfile_generator = BERT_readfromfile_generator
elif victim_model == 'BiLSTM':
    victim = VictimCache(model_path, VictimBiLSTM(model_path, task, victim_device))
    readfromfile_generator = BERT_readfromfile_generator
elif victim_model == 'surprise':
    victim = VictimCache(model_path, VictimRoBERTa(model_path, task, victim_device))
    readfromfile_generator = roberta_readfromfile_generator

# Load data
print("Loading data...")
test_dataset = Dataset.from_generator(readfromfile_generator,
                                    gen_kwargs={'subset': 'attack', 'dir': data_path, 'trim_text': True,
                                                'with_pairs': with_pairs})
if not with_pairs:
    dataset = test_dataset.map(dataset_mapping)
    dataset = dataset.remove_columns(["text"])
else:
    dataset = test_dataset.map(dataset_mapping_pairs)
    dataset = dataset.remove_columns(["text1", "text2"])

dataset = dataset.remove_columns(["fake"])

# Filter data
if using_first_n_samples:
    dataset = dataset.select(range(first_n_samples))

if targeted:
    dataset = [inst for inst in dataset if inst["y"] == 1 and victim.get_pred([inst["x"]])[0] == inst["y"]]

print("Subset size: " + str(len(dataset)))

# Prepare attack
print("Setting up the attacker...")

# Necessary to bypass the outdated SSL certifiacte on the OpenAttack servers
with no_ssl_verify():
  if using_custom_attacker:
    attacker = MyAttacker()
  else:
    filter_words = OpenAttack.attack_assist.filter_words.get_default_filter_words('english') + [SEPARATOR_CHAR]
    if attack == 'PWWS':
        attacker = OpenAttack.attackers.PWWSAttacker(token_unk=UNK_TEXT, lang='english', filter_words=filter_words)
    elif attack == 'SCPN':
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        attacker = OpenAttack.attackers.SCPNAttacker(device=attacker_device)
    elif attack == 'TextFooler':
        attacker = OpenAttack.attackers.TextFoolerAttacker(token_unk=UNK_TEXT, lang='english',
                                                           filter_words=filter_words)
    elif attack == 'DeepWordBug':
        attacker = OpenAttack.attackers.DeepWordBugAttacker(token_unk=UNK_TEXT)
    elif attack == 'VIPER':
        attacker = OpenAttack.attackers.VIPERAttacker()
    elif attack == 'GAN':
        attacker = OpenAttack.attackers.GANAttacker()
    elif attack == 'Genetic':
        attacker = OpenAttack.attackers.GeneticAttacker(lang='english', filter_words=filter_words)
    elif attack == 'PSO':
        attacker = OpenAttack.attackers.PSOAttacker(lang='english', filter_words=filter_words)
    elif attack == 'BERTattack':
        attacker = OpenAttack.attackers.BERTAttacker(filter_words=filter_words, use_bpe=False, device=attacker_device)
    elif attack == 'BAE':
        attacker = OpenAttack.attackers.BAEAttacker(device=attacker_device, filter_words=filter_words)
    else:
        attacker = None

# Run the attack
print("Evaluating the attack...")
RAW_FILE_NAME = 'raw_' + task + '_' + str(targeted) + '_' + attack + '_' + victim_model + '.tsv'
raw_path = out_dir / RAW_FILE_NAME if out_dir else None

with no_ssl_verify():
    scorer = BODEGAScore(victim_device, task, align_sentences=True, semantic_scorer="BLEURT", raw_path = raw_path)
    attack_eval = BodegaAttackEval(attacker, victim, language='english', metrics=[
        scorer  # , OpenAttack.metric.EditDistance()
    ])
    start = time.time()
    summary = attack_eval.eval_and_save_tsv(dataset, visualize=visualize_adv_examples, progress_bar=False, tsv_file_path = submission_path)
    end = time.time()
attack_time = end - start
attacker = None

# Remove unused stuff
victim.finalise()
del victim
gc.collect()
torch.cuda.empty_cache()
if "TOKENIZERS_PARALLELISM" in os.environ:
    del os.environ["TOKENIZERS_PARALLELISM"]

# Evaluate
start = time.time()
score_success, score_semantic, score_character, score_BODEGA= scorer.compute()
end = time.time()
evaluate_time = end - start

# Print results
print("Subset size: " + str(len(dataset)))
print("Success score: " + str(score_success))
print("Semantic score: " + str(score_semantic))
print("Character score: " + str(score_character))
print("BODEGA score: " + str(score_BODEGA))
print("Queries per example: " + str(summary['Avg. Victim Model Queries']))
print("Total attack time: " + str(attack_time))
print("Time per example: " + str((attack_time) / len(dataset)))
print("Total evaluation time: " + str(evaluate_time))

if out_dir:
    with open(results_path, 'w') as f:
        f.write("Subset size: " + str(len(dataset)) + '\n')
        f.write("Success score: " + str(score_success) + '\n')
        f.write("Semantic score: " + str(score_semantic) + '\n')
        f.write("Character score: " + str(score_character) + '\n')
        f.write("BODEGA score: " + str(score_BODEGA) + '\n')
        f.write("Queries per example: " + str(summary['Avg. Victim Model Queries']) + '\n')
        f.write("Total attack time: " + str(end - start) + '\n')
        f.write("Time per example: " + str((end - start) / len(dataset)) + '\n')
        f.write("Total evaluation time: " + str(evaluate_time) + '\n')

print('-')
print('Bodega metrics saved to', results_path)
print('Submission file saved to', submission_path)
