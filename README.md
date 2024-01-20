# Описание

Это реализация модели Lign-CNN для определение настоящие ли аудио перед нами или
нет.

Сама работа выполнена в виде сравнительного анализа влияния изменений модели на
ее обучение и изменение целевой метрики, которой у нас выступает Equal Error
Rate.


# Подготовка

Надо настоить окружение для python3.10 из файла requirements.txt

# Обучение

Для обучения лучшей модели достаточно запустить такую команду.

```bash
python3 train.py -c src/configs/kaggle/train.json
```

# Тест

Лучшая модель на lfcc лежит по пути lcnn-lfcc.pth

```bash
python3 test.py -c src/configs/kaggle/test.json -r lcnn-lfcc.pth
```

## Результаты

На eval датасете ошибка и порог получились такие:

```
test err: 0.05277685248662606
test thr: 0.014698578044772148
```

Прогнозы для валидационных аудио:

```
fname: aaaa-za-donbass.wav, pred: 0.912
fname: audio_1.flac, pred: 0.910
fname: audio_2.flac, pred: 0.002
fname: audio_3.flac, pred: 0.884
```