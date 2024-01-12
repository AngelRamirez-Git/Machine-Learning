"""
Este es un programa que utiliza la biblioteca spaCy para procesar un archivo de texto en español
y guardar las anotaciones en un archivo de salida.
"""
import spacy

# Cargar el modelo de procesamiento de idioma español
nlp = spacy.load('es_core_news_sm')

# Leer el contenido del archivo "pinocho.txt"
with open('pinocho.txt', 'r', encoding='utf-8') as file:
    text = file.read()

doc = nlp(text)

# Crear un archivo de salida para guardar las anotaciones
with open('pinocho_result.txt', 'w', encoding='utf-8') as output_file:
    # Iterar sobre las oraciones tokenizadas y guardar la información en el archivo
    for i, sentence in enumerate(doc.sents):
        output_file.write(f'\t\t====== Frase {i+1} tokens =======\n')

        for j, token in enumerate(sentence):
            # Justificar el número a la derecha con un ancho de 2 caracteres
            J_STR = str(j + 1).rjust(2)
            output_file.write(
                f'id: {J_STR}\t\tPalabra: {token.text.ljust(30)}\tLema: {token.lemma_}\n'
            )
