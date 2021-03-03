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
import PIL
import cv2
import numpy as np
from matplotlib import pyplot as plt

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

# TODO

# %%
#' Récupère le type de l'objet img et note que c'est bel et bien un objet de type `numpy.ndarray`,
#' cela signifie qu'en plus des fonctions opencv, tu peux aussi utiliser les fonctions numpy dessus.

# TODO

# %%
#' Récupère la `shape` de ton image, et le nombre de pixel qu'elle contient.

# TODO

# %%
#' Comme tu l'as vu précédemment l'image source est orange plutôt que bleu, en effet opencv utilise
#' par défaut le format BGR au lieu du format RGB.
#'
#' Tu sais donc convertir l'image RGB en BGR pour qu'elle s'affiche bien.

# TODO

# %%
#' - Tu sais extraire les canaux b, g et r du l'image.
#' - Tu sais afficher la composante rouge en échelle de gris grâce à matplotlib.

# TODO

# %%
#' Tu sais copier une image afin de pouvoir modifier la nouvelle image sans changer l'image de départ.

# TODO

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

# TODO

# %%
#' Tu sais réduire la taille de cette image ainsi créée à 64x64

# TODO

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

# TODO

# %%
# tmp_head_gray = np.zeros((tmp_head.shape[0], tmp_head.shape[1], 3))
#
# for i in range(3):
#     tmp_head_gray[:, :, i] = tmp_head

# TODO

# %%
#' Tu sais, avec cv2, retourner verticalement, horizontalement et transposer l'image.

# TODO

# %%
#' ### Appliquer des filtres
#'
#' Je sais appliquer un filtre gaussien (de taille 11) à l'image.

# TODO

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

# TODO

# %%
#' J'ajoute le résultat de cette détection de bord à l'image originelle. (Les bords détectés
#' apparaissent en vert sur l'image de départ)

# TODO

# %%
#' J'essaye d'utiliser la fonction `Canny` pour comparer.

# TODO

# %%
#' ## Export et fin pour OpenCV
#'
#' Je sais exporter cette belle image en utilisant cv2.

# TODO

# %%
#' ## Un  petit tour sur PIL
#'
#' J'ai compris que `PIL` utilise des objets `PIL.Image` pour gérer les images, il faut les convertir
#' en tableau `Numpy` pour avoir accès au valeurs.
#'
#' Je sais charger et afficher une image avec PIL.

# TODO

# %%
#' Je sais que PIL permet aussi retourner et transposer une image.

# TODO

# %%
#' Je sais croper une partie de l'image, la réduire et l'intégrer dans l'image originelle, je note
#' la différence de traitement qu'avec `OpenCV`.

# TODO

# %%
#' Je sais séparer les différents canaux d'une image.

# TODO

# %%
#' Je sais assombrir l'image en utilisant la méthode `.point` de l'image.

# TODO

# %%
#' Je sais appliquer un filtre de détection de bord

# TODO

# %%
#' Je sais sauvegarder cette image

# TODO

# %%
#' Je sais passer d'une image PIL à un array numpy et inversement.

# TODO

# %%
#' ### Bravo !
#'
#' Tu as finis ce notebook, il te reste des fonctions à découvrir mais tu as pu commencer à te
#' familiariser avec OpenCV et PIL !