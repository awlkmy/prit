import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from langchain.prompts import PromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import gc
import json

def request(user_message):
	emb = ""
	question = ""
	str_context = ""
	gc.collect()
	HF_TOKEN = "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"
	adapt_model_name = "D:/bibo/saiga_mistral_7b_lora/"
	base_model_name = "D:/bibo/Mistral-7B-OpenOrca/"
	tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
	tokenizer.pad_token = tokenizer.eos_token
	device_map = {"": 0}

	model = AutoPeftModelForCausalLM.from_pretrained(adapt_model_name, device_map=device_map, torch_dtype=torch.bfloat16)
	with open('context.json', 'r') as file:
		texts = json.load(file)
	vectorizer = TfidfVectorizer()
	sentence_vectors = vectorizer.fit_transform(texts)
	sentence_vectors = sentence_vectors.toarray()
	answers = []
	emb_database = torch.empty((0, 384), dtype=torch.float32)
	# Load model from HuggingFace Hub
	sent_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
	sent_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

	str_context = find_answer(q, texts, sentence_vectors, vectorizer)
	    info_prompt_less10 = PromptTemplate.from_template("user: "+str_context+"{question}\n bot: Вот ответ на ваш вопрос длиной не более 10 слов:")
	    emb = get_embedding(q)
	    print(f"CONTEXT: {str_context}")
	    if(len(answers)>0):
	      cos_sim = get_cos_sim(q)
	      max_value, max_index = torch.max(get_cos_sim(q), dim=0)
	      #print(cos_sim,max_index.item() , max_value.item())
	    if len(answers)>0 and max_value > 0.83:
	        answer = answers[max_index]
	        print(f'DATABASE: {answer}')
	    else:
	    answer = get_answer(info_prompt_less10, q)
	    emb_database = torch.cat((emb_database, emb), 0)
	    answers.append(answer)
	    print(f'MODEL: {answer}')
	    gc.collect()



def get_embedding(sentence):

    #Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Tokenize sentences
    encoded_input = sent_tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = sent_model(**encoded_input)

    # Perform pooling
    sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings

def find_answer(question, sentences, sentence_vectors, vectorizer):
    # Преобразование вопроса в вектор
    question_vector = vectorizer.transform([question]).toarray()

    # Вычисление косинусного расстояния между вопросом и каждым предложением
    similarities = cosine_similarity(question_vector, sentence_vectors)

    # Находим индекс предложения с максимальной схожестью
    max_index = np.argmax(similarities)

    # Возвращаем ответ из векторной базы данных
    return sentences[max_index]



def get_answer(info_prompt, question):

    prompt = info_prompt.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                            top_p=0.6,
                            temperature=0.4,
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=150,
                            pad_token_id=tokenizer.eos_token_id,
                            #repetition_penalty=0.6,
                            do_sample=True)

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    parsed_answer = output.split("Вот ответ на ваш вопрос длиной не более 10 слов:")[1].strip()

    if "bot:" in parsed_answer:
        parsed_answer = parsed_answer.split("bot:")[0].strip()

    # parsed_answer = output.split("bot:")[1].strip()
    return parsed_answer

def get_cos_sim(question):
    cos_sim = F.cosine_similarity(emb_database, emb, dim=1, eps=1e-8)
    return cos_sim

def multiple_question():
	questions = [
	'Что такое прокторинг?',
	'Объясни подробно, что значит аттестация?',
	'Кто может присутствовать на экзамене?',
	'Когда срок сдачи ведомостей?',
	'Сколько баллов нужно набрать на оценку "Отлично"?',
	'Где посмотреть расписание занятий?',
	'Где посмотреть адреса учебных корпусов?',
	'За что можно получить за экзамен оценку "Неудовлетворительно"?',
	'Какие основные цели определяет Положение о текущем контроле успеваемости и промежуточной аттестации обучающихся?', #определяет правила проведения текущего контроля успеваемости, промежуточной аттестации обучающихся, ликвидации академической задолженности, ведения документации по образовательным программам среднего профессионального и высшего образования, за исключением программ подготовки научно- педагогических кадров в аспирантуре.
	'На основании каких законов разработано настоящее Положение о текущем контроле успеваемости и промежуточной аттестации обучающихся?', #Настоящее Положение разработано в соответствии с: − Федеральным законом от 29.12.2012 No 273-ФЗ «Об образовании в Российской Федерации»; − приказом Министерства науки и высшего образования Российской Федерации от 06.04.2021No 245«Об утверждении Порядка организации и осуществления образовательной деятельности по образовательным программам высшего образования–программам бакалавриата, программам специалитета, программам магистратуры»; − приказом Министерства просвещения Российской Федерации от 24.08.2022No 762«Об утверждении Порядка организации и осуществления образовательной деятельности по образовательным программам среднего профессионального образования»; − письмом Министерства общего и профессионального образования Российской Федерации от 05.04.1999 No 16-52-59ин/16-13 «О рекомендациях по организации промежуточной аттестации студентов в образовательных учреждениях среднего профессионального образования»; − федеральными государственными образовательными стандартами высшего образования; − федеральными государственными образовательными стандартами среднего профессионального образования; − локальными нормативными актами ФГАОУ ВО «Тюменский государственный университет» (далее –Университет).
	'Какие структурные подразделения Университета включаются в распространение настоящего Положения о текущем контроле успеваемости и промежуточной аттестации обучающихся?', #В распространение настоящего Положения о текущем контроле успеваемости и промежуточной аттестации обучающихся включаются все структурные подразделения Университета, включая его филиалы, которые осуществляют реализацию образовательных программ среднего профессионального и высшего образования.
	'Какие особенности организации образовательного процесса учитываются при проведении текущего контроля успеваемости?', #При проведении текущего контроля успеваемости учитываются особенности организации образовательного процесса, определенные отдельными локальными нормативными актами Университета.
	'Какие основные документы определяют количество промежуточных аттестаций?', #
	'Как осуществляется фиксация информации о результатах освоения образовательной программы?',
	'Какова цель проведения промежуточной аттестации, и какие формы оценки используются для этой цели?',
	'Какие случаи могут рассматриваться как академическая задолженность?',
	'Как определяется понятие "академическая разница"?',
	'Какую основную цель преследует зачет как форма промежуточной аттестации?',
	'Какие основные аспекты оцениваются при проведении экзамена?',
	'Какие конкретные цели преследуются при проведении курсовой работы?',
	'Какие основные характеристики и цели преследуются при использовании контрольной работы?',
	'Какие основные особенности и цели связаны с проведением комплексного экзамена?',
	'Какие основные ситуации и причины могут стать основанием для подачи апелляции обучающимся?',
	'Какие основные функции выполняет зачетная книжка?',
	'Какую функцию выполняет зачетно-экзаменационная ведомость?',
	'Какие основные характеристики и признаки определяют понятие "плагиат"?',
	'Каковы основные цели и задачи текущего контроля успеваемости?',
	'Какую роль выполняют факультативные дисциплины?',
	'Каковы основные особенности учебно-экзаменационной сессии для обучающихся заочной формы обучения?',
	'Какие основные задачи и принципы лежат в основе прокторинга?',
	'Какие основные лица могут осуществлять организацию и проведение контроля успеваемости?'
	]
	
	
	
	# Find the index of the maximum value
	for q in questions:
	    print(q)
	    str_context = find_answer(q, texts, sentence_vectors, vectorizer)
	    info_prompt_less10 = PromptTemplate.from_template("user: "+str_context+"{question}\n bot: Вот ответ на ваш вопрос длиной не более 10 слов:")
	    #emb = get_embedding(q)
	    print(f"CONTEXT: {str_context}")
	    # if(len(answers)>0):
	    #   cos_sim = get_cos_sim(q)
	    #   max_value, max_index = torch.max(get_cos_sim(q), dim=0)
	    #   #print(cos_sim,max_index.item() , max_value.item())
	    # if len(answers)>0 and max_value > 0.83:
	    #     answer = answers[max_index]
	    #     print(f'DATABASE: {answer}')
	    # else:
	    answer = get_answer(info_prompt_less10, q)
	    #emb_database = torch.cat((emb_database, emb), 0)
	    answers.append(answer)
	    print(f'MODEL: {answer}')
	    gc.collect()
	    print()
