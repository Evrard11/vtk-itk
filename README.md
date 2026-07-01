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
Les visualisations produites (`show_slices`, `show_overlap_slices`) permettent d'inspecter l'alignement coupe par coupe.

La superposition des coupes axiales centrales et l'image de différence confirment visuellement l'alignement : les structures anatomiques se chevauchent correctement et les zones de désalignement se limitent aux abords de la tumeur, là où une déformation locale subsiste entre les deux acquisitions.

### 1.4 Limites

- **Grille B-spline empirique** : 8 nœuds est un choix non validé quantitativement. Trop fin,
  la B-spline déforme des structures saines mais trop grossier, elle rate les déformations locales fines.
- **Absence de masque** : la métrique est calculée sur le volume entier (fond inclus), ce qui dilue
  l'information sur les structures pertinentes.

# 2. Segmentation

La segmentation n'utilise pas les filtres ITK (region growing, etc) mais une approche seuillage + morphologie mathématique avec `scipy.ndimage`, appliquée sur les volumes convertis en tableaux numpy. Ce choix donne un contrôle plus direct sur l'enchaînement des opérations et facilite l'ajustement des paramètres.

### 2.1 Seuillage par intensité

Sur les séquences T1 utilisées ici, la tumeur apparaît hypointense (plus sombre que le parenchyme environnant). Un lissage gaussien (`sigma=2`) est appliqué en amont pour réduire le bruit, puis un double seuil `50 < intensité < 200` isole les zones sombres du corps tout en excluant le fond (intensité quasi nulle hors du crâne).

### 2.2 Nettoyage morphologique

Le seuillage seul capture aussi les autres structures fines et sombres. Une érosion (5 itérations) élimine ces structures tout en préservant la masse tumorale, plus compacte. Les composantes connexes sont ensuite labellisées (`ndi.label`), et celles qui touchent le bord du volume sont écartées (fond résiduel, artefacts en bordure de champ après recalage). Parmi les composantes restantes, la plus volumineuse est retenue comme tumeur.

Une dilatation (5 itérations) puis une fermeture morphologique (3 itérations) compensent l'érosion initiale et lissent le contour final.

### 2.3 Résultats

Les deux masques sont exportés en `data/mask_gre1.nrrd` et `data/mask_gre2.nrrd`. Le script affiche en console le volume de chaque masque ainsi que le delta entre les deux acquisitions, base du calcul des métriques utilisées en partie 3.

```
Volume GRE1 : 93741 mm3
Volume GRE2 : 104825 mm3
Delta volume : +11084 mm3
```

Le volume tumoral augmente d'environ 11.8 % entre les deux acquisitions.

### 2.4 Limites

- **Seuils empiriques** : `50` et `200` sont calibrés sur ce cas précis et ne généralisent pas à d'autres situations.
- **Hypothèse "plus grande composante non bordante"** : fonctionne ici car la tumeur est la structure sombre compacte dominante, mais échouerait en présence d'une autre masse sombre de volume comparable comme une lésion ou un artefact.
- **Pas de contrainte anatomique** : la segmentation repose uniquement sur l'intensité et la compacité géométrique.

# 3. Analyse et visualisation des changements

### 3.1 Calcul des changements

À partir des deux masques binaires, trois cartes sont calculées par opérations booléennes : `added` (présent en gre2, absent en gre1), `removed` (logique inverse) et `stable` (intersection). Ces trois matrices sont ensuite reconverties en images ITK avec les infos spatiales (spacing, origin, direction) du masque de référence.

### 3.2 Métriques

- **Volume** : nombre de voxels de chaque masque et delta.
- **Intensité moyenne** : intensité moyenne de l'image originale à l'intérieur du masque, pour chaque instant.
- **Dice** : coefficient de chevauchement `2 * |M1 ∩ M2| / (|M1| + |M2|)`, indicateur de la stabilité spatiale de la tumeur entre les deux acquisitions.

```
volume initial               : 93741 vox
volume évolution             : 104825 vox
delta des volumes            : +11084 vox
intensité moyenne initial    : 99.80
intensité moyenne évolution  : 104.84
dice                         : 0.938
```

L'intensité moyenne à l'intérieur du masque augmente elle aussi (+5.0), cohérent avec une progression tumorale plutôt qu'un simple élargissement du contour sur fond identique. Le Dice élevé (0.938) confirme que le noyau de la tumeur reste spatialement stable le même : on peut en conclure que le delta du volume correspond à une extension de la masse existante et pas à l'apparition nouvelle d'une lésion.

### 3.3 Visualisation VTK

La scène superpose quatre couches via `vtkImageSliceMapper`/`vtkImageSlice` : l'image de base, puis les masques `stable` (jaune), `added` (vert) et `removed` (rouge), chacun avec un LUT à deux entrées (transparent / couleur) pour rester lisible en superposition.

La coupe initiale est choisie automatiquement sur l'axe où la somme des voxels tumoraux (union des trois masques) est maximale, pour afficher une vue pertinente.

Une classe d'interaction ajoute :
- **Scroll** : navigation dans les coupes de l'axe actuel.
- **Clic gauche** : après une rotation de caméra, l'axe dominant de la direction de projection est recalculé et l'orientation des coupes (sagittal/coronal/axial) bascule dans la foulée.
- **Clic droit** : Change l'affichage des métriques en overlay (volume / intensité / dice).

### 3.4 Limites

- Le Dice et les volumes sont calculés sur le masque entier : en cas de lésions multiples, aucune distinction n'est faite entre elles.
- La visualisation des changements dépend directement de la qualité du recalage et de la segmentation