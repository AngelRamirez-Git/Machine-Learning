"""
Este es un programa que utiliza la biblioteca NLTK
para procesar un archivo de texto en español
y guardar las anotaciones en un archivo de salida.
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Descargar los recursos necesarios para el procesamiento en español
nltk.download('punkt')
nltk.download('stopwords')

# Configurar el stemmer para español
stemmer = SnowballStemmer('spanish')

# Leer el contenido del archivo "pinocho.txt"
with open('pinocho.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenizar el texto en palabras
words = word_tokenize(text, language='spanish')

# Lematizar las palabras
lemmas = [stemmer.stem(word) for word in words]

# Crear un archivo de salida para guardar las anotaciones
with open('pinocho_result.txt', 'w', encoding='utf-8') as output_file:
    for i, (token, lemma) in enumerate(zip(words, lemmas)):
        output_file.write(f'id: {i + 1}\t\tPalabra: {token}\tLema: {lemma}\n')
        