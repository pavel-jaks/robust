# TOSOs

## Konzultace 10. února

* [x] Sjednotit značení; provést odchaotičnění
* [x] Zjednodušit kód; zavést hot reload
* [x] Rozumět kódu
* [x] Popsat v BP to, co je v kódu
* [x] Zavést při učení NN mini-batche

## TODO - 19. 2. 2022

* [x] Srovnat SGD, Momentum a Nesterova
* [x] Provést Experiment Ivan pro 100 opakování
* [x] Zapsat, co to je testovací dataset
* [x] Zapsat, jak jsem měřil, co je lepší algoritmus - a co jsem měřil

## Konzultace 24. 2. 2022

* [x] Separovat SGD
* [x] Sjednotit značení podruhé
* [x] Rozepsat se více o test datasetu
* [x] Rozepsat se více o svých numerických experimentech
* [x] Překlady do češtiny - používat anglické výrazy

## TODO 4. 3. 2022

* [x] Očárkovat a otečkovat rovnice
* [x] Zvektorizovat grafy
* [x] Začít si s adversariálními vzorky

## TODO 6. 3. 2022

* [x] Targeted vs. untargeted
* [x] definovat útočníka a oběť
* [ ] Ověřit fenomén na str. 51 sekce A část FGSM v článku Towards Evaluating the Robustness of Neural Networks

## TODO 7. 3. 2022

* [ ] Dovymyslet citaci u MNISTu
* [ ] Zakomponovat Liho

## Konzultace 9. 3. 2022

* [x] Naimplementovat různé typy útoků
* [x] Sekci Srovnání algoritmů učení dát do samostatné kapitoly

## TODO 18. 3. 2022

* [x] Nejdříve vrstvy, potom neurony
* [x] Jednotlivé komentáře

## Konzultace 30. 3. 2022

* [x] Čárka na krát v zápisu dimenzí
* [x] Pomlčka, co vypadá jako mínus
* [x] Proč je Něstěrov výpočetně náročnější

## TODO 7. 4. 2022

* [ ] Naimplementovat line-search pořádně
* [ ] Naimplementovat L-BFGS jako L-BFGS
* [ ] Adversariální experimenty
  * [ ] Vygenerovat spoustu AE
  * [ ] Generovat pro různá $\kappa$
    * [ ] Vliv různosti $\kappa$ pro FGSM metody
  * [ ] L-BFGS - Srovnání best and worst case
    * [ ] Ideálně cross-label tabulka
    * [ ] Tabulka výsledky pro $\lambda$
  * [ ] Zkusit různé normy pro CW
  * [ ] C&W - Generovat pro různá *c*
    * [ ] Tabulka s úspěšností

## Návrh provedení

* Podívat se na line-search
* Celkově
  * Funkce pro generování
  * Fixně pro jedno $\kappa$ a maximovou normu
* FGSM Metody
  * Různá $\kappa$
  * maximová norma
* L-BFGS
  * Cross-label tabulka
  * Rozdělení vzniklých parametrů $\lambda$
  * Maximová norma
* CW
  * Úspěšnost metody v závislosti na $c$ a volbě normy

## Otázky

1. V notebooku A_ALLGEN.ipynb mi to vychází docela blbě,
  totiž u podstatné části vzorků je gradient $\nabla_x L(F_\theta (x), y) = 0$,
  což je pro FGSM metody špatné. Co s tím?
     1. Prostě říct, že ty gradienty jsou malé, a kašlat na to?
2. Jak záleží na volbě normy?
   1. V CW útoku při volbě $\|.\|_\infty$ to vypadá úplně jinak, než u CW útoku při volbě $\|.\|_1$.
3. Kterou normou to mám clipovat?

## Konzultace 13. 4. 2022

* [ ] Uklidit repo
