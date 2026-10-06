import json

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ------------------------------------------------------------
# Налаштування сторінки
# ------------------------------------------------------------
st.set_page_config(
    page_title="Класифікація тварин",
    page_icon="🐾",
    layout="wide",
)

MODEL_PATH = "animals10_cnn_classifier.keras"
CLASSES_PATH = "class_names.json"
IMG_SIZE = 64  # розмір входу нейронної мережі (64x64x3)

# Українські назви та емодзі для класів
UA_NAMES = {
    "Butterfly": "Метелик 🦋",
    "Cat": "Кіт 🐱",
    "Chicken": "Курка 🐔",
    "Cow": "Корова 🐄",
    "Dog": "Собака 🐶",
    "Elephant": "Слон 🐘",
    "Horse": "Кінь 🐴",
    "Sheep": "Вівця 🐑",
    "Spider": "Павук 🕷️",
    "Squirrel": "Білка 🐿️",
}


# ------------------------------------------------------------
# Завантаження моделі та списку класів (з кешуванням)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    """Модель завантажується один раз і зберігається в кеші Streamlit."""
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASSES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Попередня обробка зображення (так само, як під час навчання)
# ------------------------------------------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    # 1. RGB (фото можуть бути чорно-білими або з прозорістю)
    image = image.convert("RGB")
    # 2. Обрізання до квадрата по центру, щоб тварина не сплющилась
    w, h = image.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    image = image.crop((left, top, left + s, top + s))
    # 3. Зміна розміру до 64x64
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    # 4. Масив float32 і нормалізація 0-255 -> 0-1
    arr = np.asarray(image, dtype="float32") / 255.0
    # 5. Додаємо вимір батчу: (1, 64, 64, 3)
    return np.expand_dims(arr, axis=0)


def ua(name: str) -> str:
    return UA_NAMES.get(name, name)


# ------------------------------------------------------------
# Бічна панель з інформацією про модель
# ------------------------------------------------------------
class_names = load_class_names()

with st.sidebar:
    st.header("ℹ️ Про застосунок")
    st.write(
        "Застосунок визначає, яка тварина зображена на фото, "
        "за допомогою згорткової нейронної мережі (CNN)."
    )
    st.subheader("Модель")
    st.markdown(
        "- Набір даних: **Animals-10** (Hugging Face)\n"
        "- 23 554 фото, 10 класів\n"
        "- Вхід мережі: 64×64×3\n"
        "- 3 згорткові блоки + Dense\n"
        "- Точність на тесті: **≈ 80 %**"
    )
    st.subheader("Класи, які розпізнає модель")
    st.write(", ".join(ua(c) for c in class_names))

# ------------------------------------------------------------
# Основна частина
# ------------------------------------------------------------
st.title("🐾 Класифікація зображень тварин")
st.write(
    "Завантажте фото однієї з 10 тварин, і нейронна мережа визначить, хто на ньому зображений. "
    "Найкраще працюють фото, де тварина одна і займає більшу частину кадру."
)

with st.spinner("Завантаження моделі..."):
    model = load_model()

uploaded_file = st.file_uploader(
    "Оберіть зображення (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("👆 Завантажте зображення, щоб отримати прогноз.")
    st.stop()

image = Image.open(uploaded_file)

col_img, col_res = st.columns([1, 1], gap="large")

with col_img:
    st.subheader("Завантажене зображення")
    st.image(image, width="stretch")
    with st.expander("Що бачить модель (64×64)"):
        st.image(
            Image.fromarray((preprocess_image(image)[0] * 255).astype("uint8")),
            width=200,
        )

# Попередня обробка та прогноз
input_array = preprocess_image(image)
probabilities = model.predict(input_array, verbose=0)[0]
pred_idx = int(np.argmax(probabilities))
pred_name = class_names[pred_idx]
confidence = float(probabilities[pred_idx])

with col_res:
    st.subheader("Результат")
    st.metric("Модель вважає, що це", ua(pred_name), f"впевненість {confidence:.1%}")

    if confidence >= 0.7:
        st.success("Модель упевнена у своєму прогнозі.")
    elif confidence >= 0.4:
        st.warning("Модель не дуже впевнена. Спробуйте інше, чіткіше фото.")
    else:
        st.error("Модель сумнівається. Можливо, на фото тварина, якої немає серед 10 класів.")

    st.markdown("**Топ-3 найімовірніші класи:**")
    for i in np.argsort(probabilities)[::-1][:3]:
        st.write(f"{ua(class_names[i])} — {probabilities[i]:.1%}")
        st.progress(float(probabilities[i]))

st.subheader("📊 Ймовірності всіх класів")
chart_df = pd.DataFrame(
    {"Клас": [ua(c) for c in class_names], "Ймовірність": probabilities}
).set_index("Клас").sort_values("Ймовірність", ascending=False)
st.bar_chart(chart_df)
