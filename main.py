#!/usr/bin/env python
# coding: utf-8

#' # Formation OpenCV & PIL
#'
#' Ce notebook permet de s'assurer que tu maîtrises un minimum OpenCV. La computer vision est l'axe
#' principal de développement de Sicara. OpenCV est un outil implémentant des fonctions optimisées
#' de computer vision, il est donc au coeur de notre métier.
#'
#' Avant de commencer ce tutoriel, assures-toi d'être à l'aise avec Numpy. En effet, OpenCV utilise
#' des tableau Numpy pour stocker ses données. Ce notebook a fortement été inspiré par
#' https://github.com/handee/opencv-gettingstarted.
#'
#' ### Cheat sheets
#'
#' - OpenCV : https://pymotion.com/fichiers/1718/
#' - PIL : à ajouter mais la doc est bien : https://pillow.readthedocs.io/en/stable/
#'
#' Et n'oublie pas d'utiliser les raccourcis.
#'
#' -----------------
#'
#' ## Premier setup
#'
#' - Importer OpenCV (et l'installer s'il n'est pas présent : `pip install opencv-python`).
#'     - Note que ce module s'importe sous le nom `cv2`.
#'     - Ce module n'est pas le module officiel (https://pypi.org/project/opencv-python/),
#'     mais il s'installe beaucoup plus simplement que la version officielle en C.
#' - Importer Pillow (et l'installer s'il n'est pas présent : `pip install Pillow`).
#'     - Note que ce module s'import sous le nom `PIL`
#'     - PIL permet complête OpenCV.
#' - Importer Numpy (et l'installer s'il n'est pas présent : `pip install numpy`).
#' - Importer pyplot qui vient du module matplotlib (et l'installer s'il n'est pas présent : `pip install matplotlib`).
#' - Utilise l'alias `np` pour désigner le module `numpy` et `plt` pour le module `pyplot`.

# %%
from PIL import Image, ImageFilter
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.signal as sg

# %%
#' ## Charger et afficher une image
#'
#' Une des premières fonctionnalités offerte par OpenCV est de transformer une
#' image en un tableau Numpy. Tu sais trouver comment faire. Tu sais aussi l'afficher avec matplotlib.
#' Tu peux utiliser cette image : http://www.freakingnews.com/pictures/73500/The-Sandworm-Riders-73933.jpg
#'
#' L'image devrait s'afficher bleu, trouve pourquoi et corrige le problème.
#'
#' Tu peux utiliser `plt.figure(figsize=(15,15))` avant d'afficher une image pour que cette dernière
#' s'affiche mieux.

IMAGE_FILENAME = "The-Sandworm-Riders-73933.jpg"
IMAGE_PATH = os.path.join("images", IMAGE_FILENAME)

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
plt.imshow(image)
plt.show()

# %%
#' Récupère le type de l'objet img et note que c'est bel et bien un objet de type `numpy.ndarray`,
#' cela signifie qu'en plus des fonctions opencv, tu peux aussi utiliser les fonctions numpy dessus.

assert type(image) == np.ndarray

# %%
#' Récupère la `shape` de ton image, et le nombre de pixel qu'elle contient.

(height, width, _) = image.shape
number_of_pixels = height * width

# %%
#' Comme tu l'as vu précédemment l'image source est orange plutôt que bleu, en effet opencv utilise
#' par défaut le format BGR au lieu du format RGB.
#'
#' Tu sais donc convertir l'image RGB en BGR pour qu'elle s'affiche bien.

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

# %%
#' - Tu sais extraire les canaux b, g et r du l'image.
#' - Tu sais afficher la composante rouge en échelle de gris grâce à matplotlib.

image_red_channel = image[:, :, 0]
image_green_channel = image[:, :, 1]
image_blue_channel = image[:, :, 2]

plt.imshow(image_red_channel, cmap="gray")
plt.show()

# %%
#' Tu sais copier une image afin de pouvoir modifier la nouvelle image sans changer l'image de départ.
image_copy = image.copy()

# %%
#' ## Manipulations de l'image
#'
#' ### Resize, Rotate & Crop
#'
#' Tu sais isoler sur la tête du guerrier Fremen. Essaye de garder un format carré (moins de calcul
#' à faire ensuite pour toi).
#'
#' Hint : L'échelle sur l'image représente la numérotation des pixels, ça pourra t'aider. Mais rappel
#' toi que l'on travail sur une matrice.
image_cropped = image[50:150, 675:775]
plt.imshow(image_cropped)
plt.show()

# %%
#' Tu sais réduire la taille de cette image ainsi créée à 64x64
image_resized = cv2.resize(image_cropped, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
plt.imshow(image_resized)
plt.show()

# %%
#' En utilisant l'image de la tête du guerrier je sais :
#'
#' - La tourner à 90 degrés.
#' - La transformer en échelle de gris.
#'     - Tu peux utiliser `cv2.cvtColor` mais si tu le fais tu ne pourras pas directement faire
#'     l'étape suivante. Essaye de comprendre pourquoi et trouve une solution au problème.
#' - Puis la placer sur l'image entière originelle.
#'
#' Rappel toi que tu travailles avec des matrices Numpy. Toutes ces étapes sont faisable avec Numpy seul.
#'
#' Hint : https://stackoverflow.com/questions/26506204/replace-sub-part-of-matrix-by-another-small-matrix-in-numpy


image_rotated = np.rot90(image_cropped)
image_rotated = image_rotated.sum(axis=2) / 3
image_rotated = np.expand_dims(image_rotated, axis=2)
image_rotated = np.repeat(image_rotated, 3, axis=2)
image_copy[50:150, 675:775] = image_rotated
plt.imshow(image_copy)
plt.show()

# %%
#' Tu sais, avec cv2, retourner verticalement, horizontalement et transposer l'image.

vertically_flipped_image = cv2.flip(image_copy, 0)
plt.imshow(vertically_flipped_image)
plt.show()

horizontally_flipped_image = cv2.flip(image_copy, 1)
plt.imshow(horizontally_flipped_image)
plt.show()

transposed_image = cv2.transpose(image_copy)
plt.imshow(transposed_image)
plt.show()


# %%
#' ### Appliquer des filtres
#'
#' Je sais appliquer un filtre gaussien (de taille 11) à l'image.

image_smoothed = cv2.GaussianBlur(image_copy, (11, 11), 1)
plt.imshow(image_smoothed)
plt.show()

# %%
#' #### Tu sais appliquer n'importe quel filtre à une image
#'
#' - Regarde cette vidéo pour comprendre comment la détection de bord marche :
#' https://www.youtube.com/watch?v=uihBwtPIBxM
#' - Jete un coup d'oeil à cette page Wikipedia pour voir les différents type de filtres usuelles :
#' https://en.wikipedia.org/wiki/Kernel_(image_processing)
#'
#' Tu sais appliquer un filtre de détection de bord à l'image sans utiliser la fonction `Sobel` mais
#' ton ou tes propre(s) noyau(x) trouvé(s) sur Wikipedia.
#'
#' Hint :
#' - L'application d'un filtre se fait sur une matrice à deux dimensions, une conversion grayscale
#' permet de revenir sur une dimension.
#' - Tu peux utiliser `plt.figure(figsize=(15,15))` avant d'afficher une image pour que cette dernière s'affiche mieux.

image_greyscale = np.sum(image, axis=2) / 3
image_greyscale_smoothed = cv2.GaussianBlur(image_greyscale, (11, 11), 1)
edge_detection_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
image_convolved = sg.convolve(
    image_greyscale_smoothed, edge_detection_kernel, mode="same"
)
image_convolved = np.clip(image_convolved, 0, 255)
plt.imshow(image_convolved, cmap="gray")
plt.show()

# %%
#' J'ajoute le résultat de cette détection de bord à l'image originelle. (Les bords détectés
#' apparaissent en vert sur l'image de départ)

GREEN_COLOR_PIXEL = [0, 255, 0]
image_with_edges = image.copy()
image_with_edges[image_convolved > 20] = GREEN_COLOR_PIXEL

plt.imshow(image_with_edges)
plt.show()

# %%
#' J'essaye d'utiliser la fonction `Canny` pour comparer.

image_greyscale_smoothed = np.uint8(image_greyscale_smoothed)
edges = cv2.Canny(image_greyscale_smoothed, 20, 200)
image_with_canny_edges = image.copy()
image_with_canny_edges[edges >= 20] = GREEN_COLOR_PIXEL
plt.imshow(image_with_canny_edges)
plt.show()

# %%
#' ## Export et fin pour OpenCV
#'
#' Je sais exporter cette belle image en utilisant cv2.

cv2.imwrite(
    os.path.join("images", "output.png"),
    cv2.cvtColor(image_with_canny_edges, cv2.COLOR_RGB2BGR),
)

# %%
#' ## Un  petit tour sur PIL
#'
#' J'ai compris que `PIL` utilise des objets `PIL.Image` pour gérer les images, il faut les convertir
#' en tableau `Numpy` pour avoir accès au valeurs.
#'
#' Je sais charger et afficher une image avec PIL.
pil_image = Image.open(IMAGE_PATH)
pil_image.show()

# %%
#' Je sais que PIL permet aussi retourner et transposer une image.

pil_image.transpose(Image.ROTATE_180).show()
pil_image.transpose(Image.TRANSPOSE).show()

# %%
#' Je sais croper une partie de l'image, la réduire et l'intégrer dans l'image originelle, je note
#' la différence de traitement qu'avec `OpenCV`.

pil_image_cropped = pil_image.crop(
    (100, 150, 400, 450)
)  # we use a bounding box here, defined by two points
pil_image_resized = pil_image_cropped.resize((100, 100))
pil_image_copy = pil_image.copy()
paste_offset = (100, 100)
pil_image_copy.paste(pil_image_resized, paste_offset)
pil_image_copy.show()

# %%
#' Je sais séparer les différents canaux d'une image.

(pil_red_channel, pil_green_channel, pil_blue_channel) = pil_image.split()
pil_blue_channel.show()

# %%
#' Je sais assombrir l'image en utilisant la méthode `.point` de l'image.

dark_image = pil_image.point(lambda p: p > 120 and 255)
dark_image.show()


# %%
#' Je sais appliquer un filtre de détection de bord

pil_image_with_edges = pil_image.filter(ImageFilter.FIND_EDGES)
pil_image_with_edges.show()

# %%
#' Je sais sauvegarder cette image

pil_image_with_edges.save(os.path.join("images", "output_pil.jpeg"), "JPEG")

# %%
#' Je sais passer d'une image PIL à un array numpy et inversement.

numpy_array = np.array(pil_image)
pil_image = Image.fromarray(numpy_array)

# %%
#' ### Bravo !
#'
#' Tu as finis ce notebook, il te reste des fonctions à découvrir mais tu as pu commencer à te
#' familiariser avec OpenCV et PIL !
