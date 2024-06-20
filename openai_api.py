# Requirement:
#   pip install "openai<1.0"
# Usage:
#   python openai_api.py
# Visit http://localhost:8000/docs for documents.

import base64
import copy
import json
import time
from argparse import ArgumentParser
from contextlib import asynccontextmanager
from pprint import pprint
from typing import Dict, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig


class BasicAuthMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, username: str, password: str):
        super().__init__(app)
        self.required_credentials = base64.b64encode(
            f'{username}:{password}'.encode()).decode()

    async def dispatch(self, request: Request, call_next):
        authorization: str = request.headers.get('Authorization')
        if authorization:
            try:
                schema, credentials = authorization.split()
                if credentials == self.required_credentials:
                    return await call_next(request)
            except ValueError:
                pass

        headers = {'WWW-Authenticate': 'Basic'}
        return Response(status_code=401, headers=headers)


def _gc(forced: bool = False):
    global args
    if args.disable_gc and not forced:
        return

    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    _gc(forced=True)


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class ModelCard(BaseModel):
    id: str
    object: str = 'model'
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = 'owner'
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal['user', 'assistant', 'system', 'function']
    content: Optional[str]
    function_call: Optional[Dict] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal['user', 'assistant', 'system']] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[Dict]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Union[ChatMessage]
    finish_reason: Literal['stop', 'length', 'function_call']


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal['stop', 'length']]


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal['chat.completion', 'chat.completion.chunk']
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))


@app.get('/v1/models', response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id='gpt-3.5-turbo')
    return ModelList(data=[model_card])


# To work around that unpleasant leading-\n tokenization issue!
def add_extra_stop_words(stop_words):
    if stop_words:
        _stop_words = []
        _stop_words.extend(stop_words)
        for x in stop_words:
            s = x.lstrip('\n')
            if s and (s not in _stop_words):
                _stop_words.append(s)
        return _stop_words
    return stop_words


def trim_stop_words(response, stop_words):
    if stop_words:
        for stop in stop_words:
            idx = response.find(stop)
            if idx != -1:
                response = response[:idx]
    return response


TOOL_DESC = (
    '{name_for_model}: Call this tool to interact with the {name_for_human} API.'
    ' What is the {name_for_human} API useful for? {description_for_model} Parameters: {parameters}'
)

REACT_INSTRUCTION = """Answer the following questions as best you can. You have access to the following APIs:

{tools_text}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_name_text}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""

_TEXT_COMPLETION_CMD = object()


def parse_messages(messages, functions):
    if all(m.role != 'user' for m in messages):
        raise HTTPException(
            status_code=400,
            detail='Invalid request: Expecting at least one user message.',
        )

    # system设定, 与messages无关
    # ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
    # 如果未指定system_message, 或者指定的system_message为"You are a helpful assistant.", 那么system: "{REACT_INSTRUCTION}"
    # 如果人为指定system_message, 那么system: "{system_message}\n\n{REACT_INSTRUCTION}"
    messages = copy.deepcopy(messages)
    if messages[0].role == 'system':
        system = messages.pop(0).content.lstrip('\n').rstrip()
    else:
        system = 'You are a helpful assistant.'

    if functions:
        tools_text = []
        tools_name_text = []
        for func_info in functions:
            name = func_info.get('name', '')
            name_m = func_info.get('name_for_model', name)
            name_h = func_info.get('name_for_human', name)
            desc = func_info.get('description', '')
            desc_m = func_info.get('description_for_model', desc)
            tool = TOOL_DESC.format(
                name_for_model=name_m,
                name_for_human=name_h,
                # Hint: You can add the following format requirements in description:
                #   "Format the arguments as a JSON object."
                #   "Enclose the code within triple backticks (`) at the beginning and end of the code."
                description_for_model=desc_m,
                parameters=json.dumps(func_info['parameters'],
                                      ensure_ascii=False),
            )
            tools_text.append(tool)
            tools_name_text.append(name_m)
        tools_text = '\n\n'.join(tools_text)
        tools_name_text = ', '.join(tools_name_text)
        instruction = (REACT_INSTRUCTION.format(
            tools_text=tools_text,
            tools_name_text=tools_name_text,
        ).lstrip('\n').rstrip())
    else:
        instruction = ''

    messages_with_fncall = messages
    messages = []
    for m_idx, m in enumerate(messages_with_fncall):
        role, content, func_call = m.role, m.content, m.function_call
        content = content or ''
        content = content.lstrip('\n').rstrip()
        if role == 'function':
            if (len(messages) == 0) or (messages[-1].role != 'assistant'):
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting role assistant before role function.',
                )
            messages[-1].content += f'\nObservation: {content}'
            if m_idx == len(messages_with_fncall) - 1:
                # add a prefix for text completion
                messages[-1].content += '\nThought:'
        elif role == 'assistant':
            if len(messages) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=
                    'Invalid request: Expecting role user before role assistant.',
                )
            if func_call is None:
                if functions:
                    content = f'Thought: I now know the final answer.\nFinal Answer: {content}'
            else:
                f_name, f_args = func_call['name'], func_call['arguments']
                if not content.startswith('Thought:'):
                    content = f'Thought: {content}'
                content = f'{content}\nAction: {f_name}\nAction Input: {f_args}'
            if messages[-1].role == 'user':
                messages.append(
                    ChatMessage(role='assistant',
                                content=content.lstrip('\n').rstrip()))
            else:
                messages[-1].content += '\n' + content
        elif role == 'user':
            messages.append(
                ChatMessage(role='user',
                            content=content.lstrip('\n').rstrip()))
        else:
            raise HTTPException(
                status_code = 400,
                detail = f'Invalid request: Incorrect role {role}.')
    # ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑

    query = _TEXT_COMPLETION_CMD
    if messages[-1].role == 'user':
        query = messages[-1].content
        messages = messages[:-1]

    if len(messages) % 2 != 0:
        raise HTTPException(status_code=400, detail='Invalid request')

    history = []  # [(Q1, A1), (Q2, A2), ..., (Q_last_turn, A_last_turn)]
    for i in range(0, len(messages), 2):
        if messages[i].role == 'user' and messages[i + 1].role == 'assistant':
            usr_msg = messages[i].content.lstrip('\n').rstrip()
            bot_msg = messages[i + 1].content.lstrip('\n').rstrip()
            if instruction and (i == len(messages) - 2):
                usr_msg = f'{instruction}\n\nQuestion: {usr_msg}'
                instruction = ''
            history.append([usr_msg, bot_msg])
        else:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Expecting exactly one user (or function) role before every assistant role.',
            )
    if instruction:
        assert query is not _TEXT_COMPLETION_CMD
        query = f'{instruction}\n\nQuestion: {query}'
    return query, history, system


def parse_response(response):
    func_name, func_args = '', ''
    i = response.find('\nAction:')
    j = response.find('\nAction Input:')
    k = response.find('\nObservation:')
    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is omitted by the LLM,
            # because the output text may have discarded the stop word.
            response = response.rstrip() + '\nObservation:'  # Add it back.
        k = response.find('\nObservation:')
        func_name = response[i + len('\nAction:'):j].strip()
        func_args = response[j + len('\nAction Input:'):k].strip()

    if func_name:
        response = response[:i]
        t = response.find('Thought: ')
        if t >= 0:
            response = response[t + len('Thought: '):]
        response = response.strip()
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(
                role='assistant',
                content=response,
                function_call={
                    'name': func_name,
                    'arguments': func_args
                },
            ),
            finish_reason='function_call',
        )
        return choice_data

    z = response.rfind('\nFinal Answer: ')
    if z >= 0:
        response = response[z + len('\nFinal Answer: '):]
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role='assistant', content=response),
        finish_reason='stop',
    )
    return choice_data


# completion mode, not chat mode
def text_complete_last_message(history, stop_words_ids, gen_kwargs, system):
    im_start = '<|im_start|>'
    im_end = '<|im_end|>'
    prompt = f'{im_start}system\n{system}{im_end}'
    for i, (query, response) in enumerate(history):
        query = query.lstrip('\n').rstrip()
        response = response.lstrip('\n').rstrip()
        prompt += f'\n{im_start}user\n{query}{im_end}'
        prompt += f'\n{im_start}assistant\n{response}{im_end}'
    prompt = prompt[:-len(im_end)]

    _stop_words_ids = [tokenizer.encode(im_end)]
    if stop_words_ids:
        for s in stop_words_ids:
            _stop_words_ids.append(s)
    stop_words_ids = _stop_words_ids

    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(model.device)
    output = model.generate(input_ids,
                            stop_words_ids=stop_words_ids,
                            **gen_kwargs).tolist()[0]
    output = tokenizer.decode(output, errors='ignore')
    assert output.startswith(prompt)
    output = output[len(prompt):]
    output = trim_stop_words(output, ['<|endoftext|>', im_end])
    print(f'<completion>\n{prompt}\n<!-- *** -->\n{output}\n</completion>')
    return output

# class ChatCompletionResponse(BaseModel):
#     model: str
#     object: Literal["chat.completion", "chat.completion.chunk"]
#     choices: List[
#         Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
#     ]
#     created: Optional[int] = Field(default_factory=lambda: int(time.time()))
@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    # class ChatCompletionRequest(BaseModel):
    #     model: str
    #     messages: List[ChatMessage]
    #     functions: Optional[List[Dict]] = None
    #     temperature: Optional[float] = None
    #     top_p: Optional[float] = None
    #     max_length: Optional[int] = None
    #     stream: Optional[bool] = False
    #     stop: Optional[List[str]] = None

    # class ChatMessage(BaseModel):
    #     role: Literal["user", "assistant", "system", "function"]
    #     content: Optional[str]
    #     function_call: Optional[Dict] = None
    gen_kwargs = {}
    if request.top_k is not None:
        gen_kwargs['top_k'] = request.top_k
    if request.temperature is not None:
        if request.temperature < 0.01:
            gen_kwargs['top_k'] = 1  # greedy decoding
        else:
            # Not recommended. Please tune top_p instead.
            gen_kwargs['temperature'] = request.temperature
    if request.top_p is not None:
        gen_kwargs['top_p'] = request.top_p

    stop_words = add_extra_stop_words(request.stop)
    if request.functions:
        stop_words = stop_words or []
        if 'Observation:' not in stop_words:
            stop_words.append('Observation:')

    query, history, system = parse_messages(request.messages,
                                            request.functions)

    if request.stream:
        if request.functions:
            raise HTTPException(
                status_code=400,
                detail=
                'Invalid request: Function calling is not yet implemented for stream mode.',
            )
        generate = predict(query,
                           history,
                           request.model,
                           stop_words,
                           gen_kwargs,
                           system=system)
        return EventSourceResponse(generate, media_type='text/event-stream')

    stop_words_ids = [tokenizer.encode(s)
                      for s in stop_words] if stop_words else None
    if query is _TEXT_COMPLETION_CMD:
        response = text_complete_last_message(history,
                                              stop_words_ids=stop_words_ids,
                                              gen_kwargs=gen_kwargs,
                                              system=system)
    else:
        # ①
        # 调用情况:
        #     messages = [{"role": "user", "content": "你好"}]
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, temperature=0)

        # 传入 model.chat(...) 的参数:
        #     query: "你好"
        #     history: []
        #     stop_words: []
        #     stop_words_ids: None
        #     gen_kwargs: {'top_k': 1}

        # model.chat(...) 返回结果:
        #     response: "你好！有什么我能帮助你的吗？"

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "\u4f60\u597d\uff01\u6709\u4ec0\u4e48\u6211\u80fd\u5e2e\u52a9\u4f60\u7684\u5417\uff1f",
        #             "function_call": null
        #           },
        #           "finish_reason": "stop"
        #         }
        #       ],
        #       "created": 1702282068
        #     }
        #     rnt.choices[0].message.content: "你好！有什么我能帮助你的吗？"


        # ②
        # 调用情况:
        #     messages = [
        #         {"role": "user", "content": "你好"},
        #         {"role": "assistant", "content": "你好！很高兴为你提供帮助。"},
        #         {"role": "user", "content": "给我讲一个年轻人奋斗创业最终取得成功的故事。故事只能有一句话。"},
        #     ]
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, temperature=0)

        # 传入 model.chat(...) 的参数:
        #     query: "给我讲一个年轻人奋斗创业最终取得成功的故事。故事只能有一句话。"
        #     history: [
        #         [
        #             '你好',
        #             '你好！很高兴为你提供帮助。'
        #         ]
        #     ]
        #     stop_words: []
        #     stop_words_ids: None
        #     gen_kwargs: {'top_k': 1}

        # model.chat(...) 返回结果:
        #     response: "他从零开始，经历了无数的困难和挫折，但始终坚持不懈地努力，最终创办了一家成功的公司。"

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "\u4ed6\u4ece\u96f6\u5f00\u59cb\uff0c\u7ecf\u5386\u4e86\u65e0\u6570\u7684\u56f0\u96be\u548c\u632b\u6298\uff0c\u4f46\u59cb\u7ec8\u575a\u6301\u4e0d\u61c8\u5730\u52aa\u529b\uff0c\u6700\u7ec8\u521b\u529e\u4e86\u4e00\u5bb6\u6210\u529f\u7684\u516c\u53f8\u3002",
        #             "function_call": null
        #           },
        #           "finish_reason": "stop"
        #         }
        #       ],
        #       "created": 1702282071
        #     }
        #     rnt.choices[0].message.content: "他从零开始，经历了无数的困难和挫折，但始终坚持不懈地努力，最终创办了一家成功的公司。"


        # ③
        # 调用情况:
        #     messages = [{"role": "user", "content": "你好"}]
        #     functions = [
        #         {
        #             "name_for_human": "谷歌搜索",
        #             "name_for_model": "google_search",
        #             "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。"
        #                                      + " Format the arguments as a JSON object.",
        #             "parameters": [
        #                 {
        #                     "name": "search_query",
        #                     "description": "搜索关键词或短语",
        #                     "required": True,
        #                     "schema": {"type": "string"},
        #                 }
        #             ],
        #         },
        #         {
        #             "name_for_human": "文生图",
        #             "name_for_model": "image_gen",
        #             "description_for_model": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。"
        #                                      + " Format the arguments as a JSON object.",
        #             "parameters": [
        #                 {
        #                     "name": "prompt",
        #                     "description": "英文关键词，描述了希望图像具有什么内容",
        #                     "required": True,
        #                     "schema": {"type": "string"},
        #                 }
        #             ],
        #         },
        #     ]
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 model.chat(...) 的参数:
        #     query: f'{REACT_INSTRUCTION}\n\nQuestion: 你好'
        #     history: []
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # model.chat(...) 返回结果:
        #     response: "Thought: 提供的工具帮助较小，我将直接回答。\nFinal Answer: 你好！有什么我可以帮你的吗？"

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "\u4f60\u597d\uff01\u6709\u4ec0\u4e48\u6211\u53ef\u4ee5\u5e2e\u4f60\u7684\u5417\uff1f",
        #             "function_call": null
        #           },
        #           "finish_reason": "stop"
        #         }
        #       ],
        #       "created": 1702282228
        #     }
        #     rnt.choices[0].message.content: "你好！有什么我可以帮你的吗？"


        # ④
        # 调用情况:
        #     messages = [
        #         {"role": "user", "content": "你好"},
        #         {"role": "assistant", "content": "你好！很高兴见到你。有什么我可以帮忙的吗？"},
        #         {"role": "user", "content": "谁是周杰伦"},
        #     ]
        #     functions = 同③
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 model.chat(...) 的参数:
        #     query: "谁是周杰伦"
        #     history: [
        #         [
        #             f'{REACT_INSTRUCTION}\n\nQuestion: 你好',
        #             'Thought: 我会作答了。\nFinal answer: 你好！很高兴见到你。有什么我可以帮忙的吗？'
        #         ]
        #     ]
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # model.chat(...) 返回结果:
        #     response: "Thought: 我需要查找相关信息。\nAction: google_search\nAction Input: {"search_query": "周杰伦"}\nObservation:"

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "Thought: \u6211\u9700\u8981\u67e5\u627e\u76f8\u5173\u4fe1\u606f\u3002",
        #             "function_call": {
        #               "name": "google_search",
        #               "arguments": "{\"search_query\": \"\u5468\u6770\u4f26\"}"
        #             }
        #           },
        #           "finish_reason": "function_call"
        #         }
        #       ],
        #       "created": 1702282754
        #     }
        #     rnt.choices[0].message.content: "Thought: 我需要查找相关信息。"


        # ⑤
        # 调用情况:
        #     messages = [
        #         {'role': 'user', 'content': '你好'},
        #         {'role': 'assistant', 'content': '你好！很高兴见到你。有什么我可以帮忙的吗？'},
        #         {'role': 'user', 'content': '谁是周杰伦'},
        #         {'role': 'assistant', 'content': 'Thought: 我应该使用Google搜索查找相关信息。', 'function_call': {'name': 'google_search', 'arguments': '{"search_query": "周杰伦"}'}},
        #         {'role': 'function', 'name': 'google_search', 'content': 'Jay Chou is a Taiwanese singer.'}
        #     ]
        #     functions = 同③
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 text_complete_last_message(...) 的参数:
        #     query: <object object at 0x7f108959a390>
        #     history: [
        #         [
        #             '你好',
        #             'Thought: 我会作答了。\nFinal answer: 你好！很高兴见到你。有什么我可以帮忙的吗？'
        #         ],
        #         [
        #             f'{REACT_INSTRUCTION}\n\nQuestion: 谁是周杰伦',
        #             'Thought: 我应该使用Google搜索查找相关信息。\nAction: google_search\nAction Input: {"search_query": "周杰伦"}\nObservation: Jay Chou is a Taiwanese singer.\nThought:'
        #         ]
        #     ]
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # text_complete_last_message(...) 函数中模型原始的生成结果为:
        #     output: ' 我现在可以作答了。\nFinal Answer: 周杰伦是一位台湾歌手。<|im_end|><|endoftext|>'

        # text_complete_last_message(...) 返回结果:
        #     response: ' 我现在可以作答了。\nFinal Answer: 周杰伦是一位台湾歌手。'

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "\u5468\u6770\u4f26\u662f\u4e00\u4f4d\u53f0\u6e7e\u6b4c\u624b\u3002",
        #             "function_call": null
        #           },
        #           "finish_reason": "stop"
        #         }
        #       ],
        #       "created": 1702444283
        #     }
        #     rnt.choices[0].message.content: "周杰伦是一位台湾歌手。"


        # ⑥
        # 调用情况:
        #     messages = [
        #         {'role': 'user', 'content': '你好'},
        #         {'role': 'assistant', 'content': '你好！很高兴见到你。有什么我可以帮忙的吗？'},
        #         {'role': 'user', 'content': '谁是周杰伦'},
        #         {'role': 'assistant', 'content': 'Thought: 我应该使用Google搜索查找相关信息。', 'function_call': {'name': 'google_search', 'arguments': '{"search_query": "周杰伦"}'}},
        #         {'role': 'function', 'name': 'google_search', 'content': 'Jay Chou is a Taiwanese singer.'},
        #         {'role': 'assistant', 'content': '周杰伦（Jay Chou）是一位来自台湾的歌手。'},
        #         {'role': 'user', 'content': '他老婆是谁'}
        #     ]
        #     functions = 同③
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 model.chat(...) 的参数:
        #     query: "他老婆是谁"
        #     history: [
        #         [
        #             '你好',
        #             'Thought: 我会作答了。\nFinal answer: 你好！很高兴见到你。有什么我可以帮忙的吗？'
        #         ],
        #         [
        #             f'{REACT_INSTRUCTION}\n\nQuestion: 谁是周杰伦',
        #             'Thought: 我应该使用Google搜索查找相关信息。\nAction: google_search\nAction Input: {"search_query": "周杰伦"}\nObservation: Jay Chou is a Taiwanese singer.\nThought: 我会作答了。\nFinal answer: 周杰伦（Jay Chou）是一位来自台湾的歌手。'
        #         ]
        #     ]
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # model.chat(...) 返回结果:
        #     response: 'Thought: 我应该使用Google搜索查找相关信息。\nAction: google_search\nAction Input: {"search_query": "周杰伦老婆"}\nObservation:'

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "Thought: \u6211\u5e94\u8be5\u4f7f\u7528Google\u641c\u7d22\u67e5\u627e\u76f8\u5173\u4fe1\u606f\u3002",
        #             "function_call": {
        #               "name": "google_search",
        #               "arguments": "{\"search_query\": \"\u5468\u6770\u4f26\u8001\u5a46\"}"
        #             }
        #           },
        #           "finish_reason": "function_call"
        #         }
        #       ],
        #       "created": 1702365374
        #     }
        #     rnt.choices[0].message.content: "Thought: 我应该使用Google搜索查找相关信息。"


        # ⑦
        # 调用情况:
        #     messages = [
        #         {'role': 'user', 'content': '你好'},
        #         {'role': 'assistant', 'content': '你好！很高兴见到你。有什么我可以帮忙的吗？'},
        #         {'role': 'user', 'content': '谁是周杰伦'},
        #         {'role': 'assistant', 'content': 'Thought: 我应该使用Google搜索查找相关信息。', 'function_call': {'name': 'google_search', 'arguments': '{"search_query": "周杰伦"}'}},
        #         {'role': 'function', 'name': 'google_search', 'content': 'Jay Chou is a Taiwanese singer.'},
        #         {'role': 'assistant', 'content': '周杰伦（Jay Chou）是一位来自台湾的歌手。'},
        #         {'role': 'user', 'content': '他老婆是谁'},
        #         {'role': 'assistant', 'content': 'Thought: 我应该使用Google搜索查找相关信息。', 'function_call': {'name': 'google_search', 'arguments': '{"search_query": "周杰伦 老婆"}'}},
        #         {'role': 'function', 'name': 'google_search', 'content': 'Hannah Quinlivan'}
        #     ]
        #     functions = 同③
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 text_complete_last_message(...) 的参数:
        #     query: <object object at 0x7f108959a390>
        #     history: [
        #         [
        #             '你好',
        #             'Thought: 我会作答了。\nFinal answer: 你好！很高兴见到你。有什么我可以帮忙的吗？'
        #         ],
        #         [
        #             '谁是周杰伦',
        #             'Thought: 我应该使用Google搜索查找相关信息。\nAction: google_search\nAction Input: {"search_query": "周杰伦"}\nObservation: Jay Chou is a Taiwanese singer.\nThought: 我会作答了。\nFinal answer: 周杰伦（Jay Chou）是一位来自台湾的歌手。'
        #         ],
        #         [
        #             f'{REACT_INSTRUCTION}\n\nQuestion: 他老婆是谁',
        #             'Thought: 我应该使用Google搜索查找相关信息。\nAction: google_search\nAction Input: {"search_query": "周杰伦 老婆"}\nObservation: Hannah Quinlivan\nThought:'
        #         ]
        #     ]
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # text_complete_last_message(...) 函数中模型原始的生成结果为:
        #     output: ' 我会作答了。\nFinal Answer: 周杰伦的妻子是Hannah Quinlivan。<|im_end|><|endoftext|>'

        # text_complete_last_message(...) 返回结果:
        #     response: ' 我会作答了。\nFinal Answer: 周杰伦的妻子是Hannah Quinlivan。'

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "\u5468\u6770\u4f26\u7684\u59bb\u5b50\u662fHannah Quinlivan\u3002",
        #             "function_call": null
        #           },
        #           "finish_reason": "stop"
        #         }
        #       ],
        #       "created": 1702366124
        #     }
        #     rnt.choices[0].message.content: "周杰伦的妻子是Hannah Quinlivan。"


        # ⑧
        # 调用情况:
        #     messages = [
        #         {"role": "user", "content": "你好"},
        #         {"role": "assistant", "content": "你好！很高兴见到你。有什么我可以帮忙的吗？"},
        #         {'role': 'user', 'content': '给我画个可爱的小猫吧，最好是黑猫'}
        #     ]
        #     functions = 同③
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 model.chat(...) 的参数:
        #     query: "给我画个可爱的小猫吧，最好是黑猫"
        #     history: [
        #         [
        #             f'{REACT_INSTRUCTION}\n\nQuestion: 你好',
        #             'Thought: 我会作答了。\nFinal answer: 你好！很高兴见到你。有什么我可以帮忙的吗？'
        #         ]
        #     ]
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # model.chat(...) 返回结果:
        #     response: 'Thought: 我需要使用文生图API来生成一张小猫的图片。\nAction: image_gen\nAction Input: {"prompt": "a cute black cat"}\nObservation:'

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "Thought: \u6211\u9700\u8981\u4f7f\u7528\u6587\u751f\u56feAPI\u6765\u751f\u6210\u4e00\u5f20\u5c0f\u732b\u7684\u56fe\u7247\u3002",
        #             "function_call": {
        #               "name": "image_gen",
        #               "arguments": "{\"prompt\": \"a cute black cat\"}"
        #             }
        #           },
        #           "finish_reason": "function_call"
        #         }
        #       ],
        #       "created": 1702366727
        #     }
        #     rnt.choices[0].message.content: "Thought: 我需要使用文生图API来生成一张小猫的图片。"


        # ⑨
        # 调用情况:
        #     messages = [
        #         {"role": "user", "content": "你好"},
        #         {"role": "assistant", "content": "你好！很高兴见到你。有什么我可以帮忙的吗？"},
        #         {'role': 'user', 'content': '给我画个可爱的小猫吧，最好是黑猫'},
        #         {'role': 'assistant', 'content': 'Thought: 我应该使用文生图API来生成一张可爱的小猫图片。', 'function_call': {'name': 'image_gen', 'arguments': '{"prompt": "cute black cat"}'}},
        #         {'role': 'function', 'name': 'image_gen', 'content': '{"image_url": "https://image.pollinations.ai/prompt/cute%20black%20cat"}'}
        #     ]
        #     functions = 同③
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 text_complete_last_message(...) 的参数:
        #     query: <object object at 0x7f6ef4946390>
        #     history: [
        #         [
        #             '你好',
        #             'Thought: 我会作答了。\nFinal answer: 你好！很高兴见到你。有什么我可以帮忙的吗？'
        #         ],
        #         [
        #             f'{REACT_INSTRUCTION}\n\nQuestion: 给我画个可爱的小猫吧，最好是黑猫',
        #             'Thought: 我应该使用文生图API来生成一张可爱的小猫图片。\nAction: image_gen\nAction Input: {"prompt": "cute black cat"}\nObservation: {"image_url": "https://image.pollinations.ai/prompt/cute%20black%20cat"}\nThought:'
        #         ]
        #     ]
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # text_complete_last_message(...) 函数中模型原始的生成结果为:
        #     output: ' 我现在可以作答了。\nFinal Answer: 好的，我已经为您生成了一张可爱的小黑猫图片。![](https://image.pollinations.ai/prompt/cute%20black%20cat)<|im_end|><|endoftext|>'

        # text_complete_last_message(...) 返回结果:
        #     response: ' 我现在可以作答了。\nFinal Answer: 好的，我已经为您生成了一张可爱的小黑猫图片。![](https://image.pollinations.ai/prompt/cute%20black%20cat)'

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "\u597d\u7684\uff0c\u6211\u5df2\u7ecf\u4e3a\u60a8\u751f\u6210\u4e86\u4e00\u5f20\u53ef\u7231\u7684\u5c0f\u9ed1\u732b\u56fe\u7247\u3002![](https://image.pollinations.ai/prompt/cute%20black%20cat)",
        #             "function_call": null
        #           },
        #           "finish_reason": "stop"
        #         }
        #       ],
        #       "created": 1702367178
        #     }
        #     rnt.choices[0].message.content: "好的，我已经为您生成了一张可爱的小黑猫图片。![](https://image.pollinations.ai/prompt/cute%20black%20cat)"


        # ⑩
        # 调用情况:
        #     messages = [{'role': 'user', 'content': '波士顿天气如何？'}]
        #     functions = [
        #         {
        #             "name": "get_current_weather",
        #             "description": "Get the current weather in a given location.",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "location": {
        #                         "type": "string",
        #                         "description": "The city and state, e.g. San Francisco, CA",
        #                     },
        #                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        #                 },
        #                 "required": ["location"],
        #             },
        #         }
        #     ]
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 model.chat(...) 的参数:
        #     query: f'{REACT_INSTRUCTION}\n\nQuestion: 波士顿天气如何？'
        #     history: []
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # model.chat(...) 返回结果:
        #     response: 'Thought: 我需要获取波士顿的当前天气。\nAction: get_current_weather\nAction Input: {"location": "Boston, MA"}\nObservation:'

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "Thought: \u6211\u9700\u8981\u83b7\u53d6\u6ce2\u58eb\u987f\u7684\u5f53\u524d\u5929\u6c14\u3002",
        #             "function_call": {
        #               "name": "get_current_weather",
        #               "arguments": "{\"location\": \"Boston, MA\"}"
        #             }
        #           },
        #           "finish_reason": "function_call"
        #         }
        #       ],
        #       "created": 1702452887
        #     }
        #     rnt.choices[0].message.content: 'Thought: 我需要获取波士顿的当前天气。'


        # ①①
        # 调用情况:
        #     messages = [
        #         {'role': 'user', 'content': '波士顿天气如何？'},
        #         {'role': 'assistant', 'content': None, 'function_call': {'name': 'get_current_weather', 'arguments': '{"location": "Boston, MA"}'}},
        #         {'role': 'function', 'name': 'get_current_weather', 'content': '{"temperature": "22", "unit": "celsius", "description": "Sunny"}'}
        #     ]
        #     functions = 同⑩
        #     rnt = openai.ChatCompletion.create(model="Qwen", messages=messages, functions=functions, temperature=0)

        # 传入 text_complete_last_message(...) 的参数:
        #     query: <object object at 0x7f6ef4946390>
        #     history: [
        #         [
        #             f'{REACT_INSTRUCTION}\n\nQuestion: 波士顿天气如何？',
        #             'Thought: 我可以使用 get_current_weather API。\nAction: get_current_weather\nAction Input: {"location": "Boston, MA"}\nObservation: {"temperature": "22", "unit": "celsius", "description": "Sunny"}\nThought:'
        #         ]
        #     ]
        #     stop_words: ["Observation:"]
        #     stop_words_ids: [[37763, 367, 25]]
        #     gen_kwargs: {'top_k': 1}

        # text_complete_last_message(...) 函数中模型原始的生成结果为:
        #     output: ' 我现在知道波士顿的天气了。\nFinal Answer: 波士顿现在的天气是晴朗，温度为22摄氏度。<|im_end|><|endoftext|>'

        # text_complete_last_message(...) 返回结果:
        #     response: ' 我现在知道波士顿的天气了。\nFinal Answer: 波士顿现在的天气是晴朗，温度为22摄氏度。'

        # openai.ChatCompletion.create(...) 返回结果:
        #     rnt: {
        #       "model": "Qwen",
        #       "object": "chat.completion",
        #       "choices": [
        #         {
        #           "index": 0,
        #           "message": {
        #             "role": "assistant",
        #             "content": "\u6ce2\u58eb\u987f\u73b0\u5728\u7684\u5929\u6c14\u662f\u6674\u6717\uff0c\u6e29\u5ea6\u4e3a22\u6444\u6c0f\u5ea6\u3002",
        #             "function_call": null
        #           },
        #           "finish_reason": "stop"
        #         }
        #       ],
        #       "created": 1702453766
        #     }
        #     rnt.choices[0].message.content: '波士顿现在的天气是晴朗，温度为22摄氏度。'

        response, _ = model.chat(
            tokenizer,
            query,
            history=history,
            system=system,
            stop_words_ids=stop_words_ids,
            **gen_kwargs,
        )
        print('<chat>')
        pprint(history, indent=2)
        print(f'{query}\n<!-- *** -->\n{response}\n</chat>')
    _gc()

    # 将stop_word及其后面的字符串舍弃
    response = trim_stop_words(response, stop_words)
    if request.functions:
        # 如果调用openai.ChatCompletion.create时传入functions, 则视情况对response进行处理
        # 如果response中含有"\nAction:"和"\nAction Input:"字样, 则提取"\nAction:"与"\nAction Input:", "\nAction Input:"与"\nObservation:"(若response原本不含则在末尾添加一个)之间的func_name和func_args, 连同"\nAction:"之前的文本一并返回至json文件
        # 如果response中不含"\nAction:"和"\nAction Input:"字样, 但含有"\nFinal Answer: "字样, 则提取"\nFinal Answer: "后面的字符串返回
        choice_data = parse_response(response)
    else:
        # 如果调用openai.ChatCompletion.create时没传入functions, 则将response直接传入
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role='assistant', content=response),
            finish_reason='stop',
        )
        # class ChatCompletionResponseChoice(BaseModel):
        #     index: int
        #     message: ChatMessage
        #     finish_reason: Literal["stop", "length", "function_call"]

        # class ChatMessage(BaseModel):
        #     role: Literal["user", "assistant", "system", "function"]
        #     content: Optional[str]
        #     function_call: Optional[Dict] = None

        # class ChatCompletionResponse(BaseModel):
        #     model: str
        #     object: Literal["chat.completion", "chat.completion.chunk"]
        #     choices: List[
        #         Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]
        #     ]
        #     created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    return ChatCompletionResponse(model=request.model,
                                  choices=[choice_data],
                                  object='chat.completion')


def _dump_json(data: BaseModel, *args, **kwargs) -> str:
    try:
        return data.model_dump_json(*args, **kwargs)
    except AttributeError:  # pydantic<2.0.0
        return data.json(*args, **kwargs)  # noqa


async def predict(
    query: str,
    history: List[List[str]],
    model_id: str,
    stop_words: List[str],
    gen_kwargs: Dict,
    system: str,
):
    global model, tokenizer
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role='assistant'), finish_reason=None)
    chunk = ChatCompletionResponse(model=model_id,
                                   choices=[choice_data],
                                   object='chat.completion.chunk')
    yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    current_length = 0
    stop_words_ids = [tokenizer.encode(s)
                      for s in stop_words] if stop_words else None

    delay_token_num = max([len(x) for x in stop_words]) if stop_words_ids else 0
    response_generator = model.chat_stream(tokenizer,
                                           query,
                                           history=history,
                                           stop_words_ids=stop_words_ids,
                                           system=system,
                                           **gen_kwargs)
    for _new_response in response_generator:
        if len(_new_response) <= delay_token_num:
            continue
        new_response = _new_response[:-delay_token_num] if delay_token_num else _new_response

        if len(new_response) == current_length:
            continue

        new_text = new_response[current_length:]
        current_length = len(new_response)

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
        chunk = ChatCompletionResponse(model=model_id,
                                       choices=[choice_data],
                                       object='chat.completion.chunk')
        yield '{}'.format(_dump_json(chunk, exclude_unset=True))
    
    if current_length != len(_new_response):
        # Determine whether to print the delay tokens
        delayed_text = _new_response[current_length:]
        new_text = trim_stop_words(delayed_text, stop_words)
        if len(new_text) > 0:
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(content=new_text), finish_reason=None)
            chunk = ChatCompletionResponse(model=model_id,
                                        choices=[choice_data],
                                        object='chat.completion.chunk')
            yield '{}'.format(_dump_json(chunk, exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                     delta=DeltaMessage(),
                                                     finish_reason='stop')
    chunk = ChatCompletionResponse(model=model_id,
                                   choices=[choice_data],
                                   object='chat.completion.chunk')
    yield '{}'.format(_dump_json(chunk, exclude_unset=True))
    yield '[DONE]'

    _gc()


def _get_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-c',
        '--checkpoint-path',
        type=str,
        default='Qwen/Qwen-7B-Chat',
        help='Checkpoint name or path, default to %(default)r',
    )
    parser.add_argument('--api-auth', help='API authentication credentials')
    parser.add_argument('--cpu-only',
                        action='store_true',
                        help='Run demo with CPU only')
    parser.add_argument('--server-port',
                        type=int,
                        default=8000,
                        help='Demo server port.')
    parser.add_argument(
        '--server-name',
        type=str,
        default='127.0.0.1',
        help=
        'Demo server name. Default: 127.0.0.1, which is only visible from the local computer.'
        ' If you want other computers to access your server, use 0.0.0.0 instead.',
    )
    parser.add_argument(
        '--disable-gc',
        action='store_true',
        help='Disable GC after each response generated.',
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    if args.api_auth:
        app.add_middleware(BasicAuthMiddleware,
                           username=args.api_auth.split(':')[0],
                           password=args.api_auth.split(':')[1])

    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    model.generation_config = GenerationConfig.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        resume_download=True,
    )

    uvicorn.run(app, host=args.server_name, port=args.server_port, workers=1)
