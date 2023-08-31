# character-tokenizer

Это специальный **посимвольный** токенизатор, совместимый с библиотекой Hugging Face Transformers.
Он сделан для использования декодерными causal LM'ками. В частности, он применен при разработке модели [inkoziev/charllama-35M](https://huggingface.co/inkoziev/charllama-35M).

## Установка

```
pip install git+https://github.com/Koziev/character-tokenizer
```

## Пример использования

```py
import charactertokenizer

tokenizer = charactertokenizer.CharacterTokenizer.from_pretrained('inkoziev/charllama-35M')

prompt = '<s>У Лукоморья дуб зеленый\n'
encoded_prompt = tokenizer.encode(prompt, return_tensors='pt')
print(' | '.join(tokenizer.decode([t]) for t in encoded_prompt[0]))
```

Вывод будет таким:

```
<s> | У |   | Л | у | к | о | м | о | р | ь | я |   | д | у | б |   | з | е | л | е | н | ы | й |
```
