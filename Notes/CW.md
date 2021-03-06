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

Prvně zmíním, že hledání adv. vzorku, tedy optimalizaci výše uvedené funkce, provádím sign gradient descentem s krokem $10^{-2}$ a $100$ iterací.

Všiml jsem si zajímavé věci: Jelikož chci, aby podle mojí definice byl adv. vzorek adv. vzorkem, tedy i aby $\tilde{x}$ byl v $\kappa$-okolí $x$ (v $L^\infty$ normě), tak to někde musím clipnout.
Když to clipuju v každé iteraci, tak mi pěkně funguje útok v $L^\infty$ normě a i v $L^1$ normě, ale útok v $L^2$ normě si ani neškrtne.
Když to clipnu jednou - na konci optimalizace, tak mi zas nefunguje útok v $L^\infty$ normě, zbylé dva jsou v pohodě.
Tady možná narážím na to, že pořádně nevím, co je *box-constrained optimalizace*.

## Co je to to *c* v předpisu CW?

To je konstanta.
Lze jí štelovat a hledat minimální takovou, pro kterou neuronka misklasifikuje, nebo jí položit rovno $1$ - tahle hodnota se mi osvědčila.

## Výsledky experimentů

### c_lambda = 1

Typ CW útoku | Úspěšnost
---|---
cw_linfty_always| 74 %
cw_lone_always| 50 %
cw_ltwo_always| 8 %
cw_linfty_once| 0 %
cw_lone_once| 33 %
cw_ltwo_once| 87 %

### c_lambda = 10

Typ CW útoku | Úspěšnost
---|---
cw_linfty_always| 74 %
cw_lone_always| 87 %
cw_ltwo_always| 14 %
cw_linfty_once| 0 %
cw_lone_once| 89 %
cw_ltwo_once| 94 %

### c_lambda = 0.1

Typ CW útoku | Úspěšnost
---|---
cw_linfty_always| 74 %
cw_lone_always| 0 %
cw_ltwo_always| 1 %
cw_linfty_once| 0 %
cw_lone_once| 0 %
cw_ltwo_once| 8 %

### c_lambda = 100

Typ CW útoku | Úspěšnost
---|---
cw_linfty_always| 74 %
cw_lone_always| 90 %
cw_ltwo_always| 16 %
cw_linfty_once| 0 %
cw_lone_once| 93 %
cw_ltwo_once| 94 %
