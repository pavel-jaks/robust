# CW útok a s ním spojená pozorování

Mějme vzorek $x$ a k němu příslušnou značku $y$.
Mějme dále $\kappa$, které určuje poloměr koule, ve které hledáme adversariální vzorek v maximové normě.
Pro konkrétnost: nť je $\kappa = 50 / 255$,
tj. dovolujeme změnu v obrázku každému pixelu o 50 bodů.

Potom lze definovat CW útok jako následující optimalizační úlohu:

$\tilde{x} = \operatorname{argmin}_{\hat{x}} \|\hat{x} - x\| - c \cdot L(F(\hat{x}), y)$,

kde $F$ je neuronová síť a $L$ je ztrátová funke,
kterou byla daná síť natrénovaná.
Dále požadujeme, aby $\|\tilde{x} - x\| \leq K \cdot \kappa$, kde $K$ je konstanta zaručující, že $\|\tilde{x} - x\|_\infty \leq \kappa$, přičemž $K$ existuje, protože jsme na prostorech konečné dimenze.

Snadno lze ověřit, že pro MNIST (obrázky rozměru $28 \times 28$)
$\|\tilde{x} - x\|_2 \leq 28 \cdot \|\tilde{x} - x\|_\infty$
a
$\|\tilde{x} - x\|_1 \leq 28 \cdot 28 \cdot \|\tilde{x} - x\|_\infty$

Proto v následující sekci má smysl uvažovat různé poloměry kulových okolí, kde adv. vzorek hledáme, a to podle užité normy.

## Volba normy

Volba normy hraje roli nejen při volbě poloměru, kde adv. vzorek hledáme, ale i v kvalitativním výsledku.
Ve zkratce:

- $\|.\|_\infty$ dává kvalitativně stejné obrázky jako metody založené na FGSM
  - Tj. obrázek CW vypadá by-oko stejně jako PGD obrázek - je tam bordel v okolí číslovky
- $\|.\|_1$ dává esteticky hezčí obrázky - není tam bordel okolo číslovky.
  - Tadyten útok jenom by-oko umazává pixely, nekreslí okolo.
- $\|.\|_2$ dává něco mezi
  - Jednak umazává pixely, ale i kreslí okolo. Ale je třeba upozornit, že to kreslení okolo vypadá sofistikovaně, totiž třeba k jedničce přikreslí vodorovnou čáru nahoře, jako by tam měla být sedmička, nedělá to bordel.

## Clipování při hledání vzorku

TODO:
