# Projet VTK-ITK

**auteurs:** evrard.casamayou, gabin.clerbout, elie.dalmas

## Lancer le projet
Prérequis: python
```
py -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Objectif
L'objectif de ce projet est de réaliser le suivi longitudinal de l'évolution d'une tumeur à partir de deux scans IRM (cas `case6_gre1.nrrd` et `case6_gre2.nrrd`) effectués sur le même patient à des instants différents.

Le pipeline complet comprend :
- **Partie 1** : Recalage automatique des deux volumes (ITK)
- **Partie 2** : Segmentation des tumeurs (ITK)
- **Partie 3** : Analyse et visualisation des changements (VTK)

# 1. Recalage
### 1.1 Sous-échantillonnage
Les images sont réduites d'un facteur 3 dans chaque dimension via `BinShrinkImageFilter`,
qui moyenne les voxels avant réduction (contrairement à `ShrinkImageFilter` qui sous-échantillonne
directement et produit des artefacts d'aliasing). Cela divise le nombre de voxels par 27 et
accélère drastiquement le recalage, au prix d'une légère perte de détail fin (acceptable pour
l'alignement global).

### 1.2 Pipeline de recalage
Le recalage se fait en trois étapes progressives, chacune initialisant la suivante:
```
Recalage Rigide  →  Recalage Affine  →  Recalage B-spline
 (6 degrés)          (12 degrés)         (déformable local)
```
#### Recalage rigide

`VersorRigid3DTransform` - 3 rotations + 3 translations. L'initialisation centre la
transformation sur le centre géométrique des images pour faciliter la rotation (`CenteredTransformInitializer`, `GeometryOn`).

Les scales de l'optimiseur sont différenciées : `1.0` pour les rotations (en rad) et `0.01`
pour les translations (en mm), car 1 rad correspond approximativement à un déplacement de
~100 mm en périphérie du volume.

> `SetNumberOfLevels(1)` est utilisé ici pour des raisons de temps de calcul. En production,
> 3 niveaux multi-résolution amélioreraient la robustesse.

#### Recalage affine

`AffineTransform` (12 DoF) - ajoute cisaillement et facteurs d'échelle différentiels par axe,
utile si la résolution de coupe varie légèrement entre les deux acquisitions. Il est initialisé
(**warm-start**) depuis la transformation rigide avec le centre, la matrice et la translation qui sont
transmis directement. Cela permet d'accélèrer la convergence et réduit le risque de minimum local.

#### Recalage B-spline

`BSplineTransform` (ordre 3, grille 8 nœuds) - corrige les déformations locales (évolution
tumorale, repositionnement des organes voisins). La transformation affine est fixée comme
`MovingInitialTransform` et **n'est pas réoptimisée** : seule la grille B-spline est ajustée,
ce qui découple corrections globale et locale.

L'optimiseur `LBFGSBOptimizerv4` est choisi pour sa convergence rapide en haute dimension
(~375 paramètres pour une grille $5^3*3$) grâce à l'approximation quasi-Newton.
Les bornes sont désactivées (`bound_select = 0`), laissant les déplacements libres.

Le résultat est une `CompositeTransform` (affine + B-spline) appliquée en une passe lors du
rééchantillonnage final par `ResampleImageFilter` (interpolation linéaire).

### 1.3 Résultats
```
RMSE avant recalage : ~0.114
RMSE après recalage : ~0.074
```

La superposition des coupes axiales centrales et l'image de différence confirment visuellement l'alignement : les structures anatomiques se chevauchent correctement et les zones de désalignement se limitent aux abords de la tumeur, là où une déformation locale subsiste entre les deux acquisitions.

### 1.4 Limites

- **Grille B-spline empirique** : 8 nœuds est un choix non validé quantitativement. Trop fin,
  la B-spline déforme des structures saines mais trop grossier, elle rate les déformations locales fines.
- **Absence de masque** : la métrique est calculée sur le volume entier (fond inclus), ce qui dilue
  l'information sur les structures pertinentes.
