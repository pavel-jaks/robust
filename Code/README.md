# Code

## Structure

* Jupyter notebooks
  * A_ adversariální experimenty
  * T_ trénovací experimenty
  * R_ experimenty robustního učení
* .py soubory
  * models.py
    * definice používaných modelů
    * statická třída ModelManager
  * training.py
    * statická třída Coach
      * pro trénování neuronových sítí
  * utils.py
    * třídy pro práci s datasety
  * adversarials.py
    * obsahuje statickou třídu s metodou pro nalezaní adversarialních vzorků
* .pdf, .txt, .json
  * výsledky experimentů
