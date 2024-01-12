"""
Este es un programa que utiliza la biblioteca stanza para procesar un archivo de texto en español
y guardar las anotaciones en un archivo de salida.
"""

import stanza

# Cargar el modelo de procesamiento de idioma español
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')

# Leer el contenido del archivo "pinocho.txt"
with open('pinocho.txt', 'r', encoding='utf-8') as file:
    text = file.read()

doc = nlp(text)

# Crear un archivo de salida para guardar las anotaciones
with open('pinocho_result.txt', 'w', encoding='utf-8') as output_file:
    # Iterar sobre las oraciones tokenizadas y guardar la información en el archivo
    for i, sentence in enumerate(doc.sentences):
        output_file.write(f'\t\t====== Frase {i+1} tokens =======\n')

        for j, word in enumerate(sentence.words):
            # Justificar el número a la derecha con un ancho de 2 caracteres
            J_STR = str(j + 1).rjust(2)  # Cambio de j_str a J_STR
            output_file.write(
                f'id: {J_STR}\t\tPalabra: {word.text.ljust(30)}\tLema: {word.lemma}\n'
            )
