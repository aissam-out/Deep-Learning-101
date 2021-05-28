from googletrans import Translator


translator = Translator()
text_1 = 'bonjour tout le monde !'
text_2 = "이 문장은 한글로 쓰여졌습니다."

# translation (default destination is english)
translation_1 = translator.translate(text_1)
print(translation_1.origin, ' -> ', translation_1.text)

# language detection
t1 = translator.detect(text_2)
print(t1)
translation_2 = translator.translate(text_2)
print(translation_2.text)
