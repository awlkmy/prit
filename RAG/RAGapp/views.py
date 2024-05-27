from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views.decorators.http import require_POST
from django.conf import settings
from django.contrib.staticfiles import finders
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
import os


HF_TOKEN = None
adapt_model_name = None
base_model_name = None
tokenizer = None
device_map = None
model = None
texts = None


def load_model():
    global HF_TOKEN, adapt_model_name, base_model_name, tokenizer, device_map, model, texts

    HF_TOKEN = "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"
    adapt_model_name = os.path.join(settings.BASE_DIR, 'static', 'saiga_mistral_7b_lora')
    base_model_name = os.path.join(settings.BASE_DIR, 'static', 'Mistral-7B-OpenOrca')
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    device_map = {"": 0}
    model = AutoPeftModelForCausalLM.from_pretrained(adapt_model_name, device_map=device_map,
                                                     torch_dtype=torch.bfloat16)
    with open(os.path.join(settings.BASE_DIR, 'static', 'context.json'), 'r', encoding="UTF-8") as file:
        texts = json.load(file)


def question_answer(request):
    load_model()
    return render(request, 'main.html')


@require_POST
def ajax_response(request):
    user_message = request.POST.get('user-message')
    bot_message = bot_request(user_message)
    data = {
        'message': bot_message
    }
    print(bot_message)
    return JsonResponse(data)


def bot_request(user_message):
    emb = ""
    question = ""
    str_context = ""
    gc.collect()

    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(texts)
    sentence_vectors = sentence_vectors.toarray()
    q = user_message
    answers = []
    emb_database = torch.empty((0, 384), dtype=torch.float32)
    # Load model from HuggingFace Hub
    sent_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    sent_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    str_context = find_answer(q, texts, sentence_vectors, vectorizer)
    info_prompt_less10 = PromptTemplate.from_template(
        "user: " + str_context + "{question}\n bot: Вот ответ на ваш вопрос длиной не более 10 слов:")
    emb = get_embedding(q, sent_tokenizer, sent_model)  # Передаем sent_tokenizer в функцию get_embedding
    print(f"CONTEXT: {str_context}")
    if len(answers) > 0:
        cos_sim = get_cos_sim(q)
        max_value, max_index = torch.max(get_cos_sim(q, emb_database, emb), dim=0)
        # print(cos_sim,max_index.item() , max_value.item())
    if len(answers) > 0 and max_value > 0.83:
        answer = answers[max_index]
        print(f'DATABASE: {answer}')
    else:
        answer = get_answer(info_prompt_less10, q, tokenizer, model)
    emb_database = torch.cat((emb_database, emb), 0)
    answers.append(answer)
    print(f'MODEL: {answer}')
    gc.collect()

    return answer


def get_embedding(sentence, sent_tokenizer, sent_model):
    # Mean Pooling - Take attention mask into account for correct averaging
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
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


def get_answer(info_prompt, question, tokenizer, model):
    prompt = info_prompt.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                             top_p=0.6,
                             temperature=0.4,
                             attention_mask=inputs["attention_mask"],
                             max_new_tokens=150,
                             pad_token_id=tokenizer.eos_token_id,
                             # repetition_penalty=0.6,
                             do_sample=True)

    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    parsed_answer = output.split("Вот ответ на ваш вопрос длиной не более 10 слов:")[1].strip()

    if "bot:" in parsed_answer:
        parsed_answer = parsed_answer.split("bot:")[0].strip()

    # parsed_answer = output.split("bot:")[1].strip()
    return parsed_answer


def get_cos_sim(question, emb_database, emb):
    cos_sim = F.cosine_similarity(emb_database, emb, dim=1, eps=1e-8)
    return cos_sim


# def multiple_question():
#     questions = []
#
#     # Find the index of the maximum value
#     for q in questions:
#         print(q)
#         str_context = find_answer(q, texts, sentence_vectors, vectorizer)
#         info_prompt_less10 = PromptTemplate.from_template(
#             "user: " + str_context + "{question}\n bot: Вот ответ на ваш вопрос длиной не более 10 слов:")
#         # emb = get_embedding(q)
#         print(f"CONTEXT: {str_context}")
#         # if(len(answers)>0):
#         #   cos_sim = get_cos_sim(q)
#         #   max_value, max_index = torch.max(get_cos_sim(q), dim=0)
#         #   #print(cos_sim,max_index.item() , max_value.item())
#         # if len(answers)>0 and max_value > 0.83:
#         #     answer = answers[max_index]
#         #     print(f'DATABASE: {answer}')
#         # else:
#         answer = get_answer(info_prompt_less10, q)
#         # emb_database = torch.cat((emb_database, emb), 0)
#         answers.append(answer)
#         print(f'MODEL: {answer}')
#         gc.collect()
#         print()
