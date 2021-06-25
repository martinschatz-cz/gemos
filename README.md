Projekt GEMOS
=============
Struktura projektu
----------
### Struktura adresářů:
* data - adresář pro obrázky (obsah přidejte do gitignore)
* libs - adresář pro externí knihovny a řešení co budem testovat (typicky klon github repo)
* src - námi vytvořené zdrojové kódy
* tests - adresář pro testy

### Soubory:
* README.md - popis projektu, klidně rozšiřujte
* TODO.md - sem pište zásadnější TODO vetšího rozsahu, než to co se dá vyřešit pomocí #TODO :)
* requirements.txt - sem pište i číslo verze jakýhokoliv package co použijete


Konvence atp.
--------------
* preferoval bych kdybysme používali v kódu angl. názvy proměnných atp.
* psaní kódu, pojmenovávání proměnných atp. bych preferoval dle PEP 8 (https://www.python.org/dev/peps/pep-0008/z)
* každá funkce by měla obsahovat docstring, jinak se v tom časem utopíme
* zatim píšeme testovací kód, tak se asi nemusíme obtěžovat ošetřováním všech výjimek a generování unit testů ke všem 
funkcím

Jak to používat
-----------
V src je ct_evaluation.py. Tenhle skript slouží k měření času, který je potřeba ke zpracování jednoho obrázku. Výsledky
jsou pro vizuální kontrolu a porovnání zobrazovány v oknech.

