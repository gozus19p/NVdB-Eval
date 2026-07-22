## Istruzioni

Ti viene fornito un lemma, una parola o locuzione che rappresenta un concetto.
Il tuo compito è elencare tutti i sensi di questo lemma, riportando la glossa e almeno un esempio d'uso per ognuno.
Rispondi in JSON seguendo lo schema seguente:

```json
[
    {{
        "glossa": "La glossa del senso",
        "esempi": [
            "Esempio d'uso 1",
            "Esempio d'uso 2"
        ]
    }}
]
```

## Esempio

Lemma: "insultare"
Risposta:
```json
[
    {{
        "glossa": "offendere gravemente con parole o atti ingiuriosi o sprezzanti",
        "esempi": [
            "i. un amico",
            "i. la memoria",
            "il buon nome di qcn."
        ]
    }},
    {{
        "glossa": "assalire, aggredire",
        "esempi": [
            "i. un nemico"
        ]
    }},
    {{
        "glossa": "recare offesa, ingiuria: Boccaccio)",
        "esempi": [
            "contra i poeti tumultuosamente insultano",
            "Non tollero che mi insultino in questo modo"
        ]
    }},
    {{
        "glossa": "saltare, salire sopra",
        "esempi": [
            "i. un ostacolo"
        ]
    }}
]
```

## Input

Analizza il seguente lemma e fornisci la risposta seguendo le istruzioni sopra:
Lemma: "{lemma}"