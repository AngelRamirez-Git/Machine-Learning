"""
Este es un programa que utiliza la biblioteca scikitlearn 
para procesar un archivo de texto en español
y guardar las anotaciones en un archivo de salida.
"""
from sklearn.feature_extraction.text import CountVectorizer

# Leer el contenido del archivo "pinocho.txt"
with open('pinocho.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenizar el texto utilizando CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform([text])

# Obtener las palabras tokenizadas
tokens = vectorizer.get_feature_names_out()

# Crear un archivo de salida para guardar las anotaciones
with open('pinocho_result.txt', 'w', encoding='utf-8') as output_file:
    for i, token in enumerate(tokens):
        output_file.write(f'id: {i + 1}\t\tPalabra: {token}\n')
