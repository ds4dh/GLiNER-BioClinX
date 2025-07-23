import ast
import json
import time
import re
from langdetect import detect, DetectorFactory
from typing import List, Dict
from scholarly import scholarly


class ScholarsLLMScraper:
    def __init__(self, articles_per_query: int = 50, lang: str = 'uk'):
        self.articles_per_query = articles_per_query
        self.lang2extract = lang

    def create_chat(self, prompt):
        return prompt

    def get_prompt(self):
        messages = [
            {
                "role": "system",
                "content": "You are accurate and precise entity recognition model that extracts named entities from arbitrary text.",
            },
            {"role": "user", "content": self.prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return prompt

    def generate_responce(self, chat):
        response = self.llm.generate(chat, self.sampling_params, use_tqdm=True)
        return response

    def extract_list_from_response(self, response: List):
        s = response[0].outputs[0].text.strip()
        last_comma = s.rfind(',')
        if last_comma == -1:
            return []
        trimmed = s[:last_comma]
        fixed = trimmed + "]]"
        parsed = ast.literal_eval(fixed)
        inner_list = parsed[0]
        seen = set()
        result = []
        for item in inner_list:
            if item not in seen:
                seen.add(item)
                result.append(item)    
        return result

    def generate_queries(self):
        prompt = self.get_prompt()
        chat = self.create_chat(prompt)
        response = self.generate_responce(chat)
        parsed_list = self.extract_list_from_response(response)
        return parsed_list

    def extract_abstracts(self, queries: List[str]):
        abstracts = []
        seen_texts = set()
        for query in queries:
            search_results = scholarly.search_pubs(query)
            count = 0
            while count < self.articles_per_query:
                try:
                    publication = next(search_results)
                except StopIteration:
                    break
                abstract = publication['bib'].get('abstract', '').strip()
                if not abstract:
                    continue
                try:
                    lang = detect(abstract)
                except Exception:
                    continue
                if lang == self.lang2extract and abstract not in seen_texts:
                    abstracts.append({'text': abstract})
                    seen_texts.add(abstract)
                    count += 1
                print(count)
            with open("abstracts_check_4.json", 'w', encoding='utf-8') as f:
                json.dump(abstracts, f, ensure_ascii=False, indent=2)
            time.sleep(60*1)
        return abstracts

    def save(self, abstracts: List[Dict]):
        with open("abstracts.json", 'w', encoding='utf-8') as f:
            json.dump(abstracts, f, ensure_ascii=False, indent=2)

    def scrap(self):
        queries = [
            ' "імунна толерантність" ',
            ' "РНК-інтерференція" ',
            ' "фармакодинаміка" ',
            ' "клінічна фармакологія" ',
            ' "експресія генів" ',
            ' "діабет другого типу" ',
            ' "міграція лейкоцитів" ',
            ' "пухлини мозку" ',
            ' "онкогени" ',
            ' "нейродегенеративні хвороби" ',
            ' "кіназні каскади" ',
            ' "нейротрансмітери" ',
            ' "депресивні розлади" ',
            ' "вакцинація від COVID-19" ',
            ' "електроенцефалографія" ',
            ' "склероз" ',
            ' "CRISPR-Cas9 технологія" ',
            ' "антиген-презентувальні клітини" ',
            ' "неонатальна медицина" ',
            ' "ендокринна регуляція" ',
            ' "вакцинація" ',
            ' "онкологічна терапія" ',
            ' "анемія" ',
            ' "педіатрична онкологія" ',
            ' "алергічні реакції" ',
            ' "гормональна регуляція" ',
            ' "клінічні випробування" ',
            ' "апоптоз клітин" ',
            ' "нейропатологія" ',
            ' "гепатити" ',
            ' "біологічна терапія" ',
            ' "мікробіом кишечника" ',
            ' "хронічна втома" ',
            ' "інсулінова терапія" ',
            ' "дисліпідемія" ',
            ' "онкологія" ',
            ' "запальні цитокіни" ',
            ' "системні захворювання сполучної тканини" ',
            ' "мітохондрії" ',
            ' "гемостаз" ',
            ' "патогенез захворювань" ',
            ' "ретровіруси" ',
            ' "тканинна регенерація" ',
            ' "дисфункція органів" ',
            ' "глікемічний контроль" ',
            ' "тривожні розлади" '
        ]
        abstracts = self.extract_abstracts(queries)
        self.save(abstracts)
        return abstracts


scraper = ScholarsLLMScraper(
    articles_per_query=10,
    lang='uk'
)
abstracts = scraper.scrap()
print(len(abstracts))