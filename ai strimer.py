from PyCharacterAI import Client
import asyncio
from translate import Translator
import socket
import torch
import sounddevice as sd
import time
import os


async def main():
    # parser
    server = 'irc.chat.twitch.tv'
    port = 6667
    nickname = ''  # your twitch nickname
    token = ''  # twitch token
    channel = ''  # trackable channel
    sock = socket.socket()
    sock.connect((server, port))
    sock.send(f"PASS {token}\n".encode('utf-8'))
    sock.send(f"NICK {nickname}\n".encode('utf-8'))
    sock.send(f"JOIN {channel}\n".encode('utf-8'))

    # character
    char = ''
    token = ''
    client = Client()
    await client.authenticate_with_token(token)
    character_id = char
    chat = await client.create_or_continue_chat(character_id)

    # voice
    local_file = 'model.pt'
    language = 'ru'
    model_id = 'v4_ru'
    device = torch.device('cuda')

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                       local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)
    sample_rate = 48000
    speaker = 'xenia'
    put_accent = True
    put_yo = True

    while True:
        resp = sock.recv(2048).decode('utf-8')

        if resp.startswith('PING'):
            sock.send("PONG\n".encode('utf-8'))

        if len(str(resp).split('PRIVMSG #nirayae :')) > 1:
            message = str(resp).split('PRIVMSG #nirayae :')[-1].rstrip()
            print(message)
            answer = await chat.send_message(message)
            translator = Translator(to_lang="ru")
            translation_text = translator.translate(answer.text)
            print(f"{answer.src_character_name}: {translation_text}")
            lst = []
            lst2 = []
            flag = False
            k = 0

            for el in translation_text:
                if el == '*':
                    flag = True
                    k += 1
                    continue
                if k % 2 == 0:
                    flag = False
                    k = 0
                if flag:
                    lst2.append(el)
                else:
                    lst.append(el)
            text = f"{message}. {''.join(lst)[1:]}"
            audio = model.apply_tts(text=text,
                                    speaker=speaker,
                                    sample_rate=sample_rate,
                                    put_accent=put_accent,
                                    put_yo=put_yo)
            sd.play(audio, sample_rate)
            time.sleep((len(audio) / sample_rate) + 0.1)
            sd.stop


asyncio.run(main())
