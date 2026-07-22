## Istruzioni

Ti vengono fornite due liste di sensi per un lemma, una lista è quella estratta da un dizionario e l'altra è quella estratta da un modello di linguaggio.
Il tuo compito è giudicare il grado di allineamento dei sensi, ovvero se i sensi estratti dal modello di linguaggio corrispondono a quelli estratti dal dizionario.

Nel processo di analisi, segui le definizioni di "true positive", "false positive" e "false negative" riportate di seguito:
- "true positive": un senso estratto dal modello di linguaggio che corrisponde a un senso estratto dal dizionario, a prescindere dalla formulazione della glossa.
- "false positive": un senso estratto dal modello di linguaggio che non corrisponde a nessun senso estratto dal dizionario.
- "false negative": un senso estratto dal dizionario che non corrisponde a nessun senso estratto dal modello di linguaggio.

Rispondi in JSON, elencando gli indici dei true positive, dei false positive e dei false negative, seguendo lo schema seguente:
```json
{{
    "true_positive": [
        {{
            "ground_truth": 1, "prediction": 1
        }},
        {{
            "ground_truth": 2, "prediction": 2
        }}
    ],
    "false_positive": [
        {{
            "ground_truth": null, "prediction": 4
        }}
    ],
    "false_negative": [
        {{
            "ground_truth": 3, "prediction": null
        }}
    ]
}}
```

## Esempio

Lista di sensi estratti da un dizionario:
1. offendere gravemente con parole o atti ingiuriosi o sprezzanti
2. assalire, aggredire
3. recare offesa, ingiuria: Boccaccio)
4. saltare, salire sopra

Lista di sensi estratti da un modello di linguaggio:
1. denigrare, offendere con parole o atti ingiuriosi o sprezzanti
2. aggredire
3. saltare, salire sopra
4. sussultare

Risposta:

```json
{{
    "true_positive": [
        {{
            "ground_truth": 1,
            "prediction": 1
        }},
        {{
            "ground_truth": 2,
            "prediction": 2
        }}
    ],
    "false_positive": [
        {{
            "ground_truth": null,
            "prediction": 4
        }}
    ],
    "false_negative": [
        {{
            "ground_truth": 3,
            "prediction": null
        }}
    ]
}}
```

## Input

Analizza le seguenti liste di sensi e fornisci la risposta seguendo le istruzioni sopra:
Lista di sensi estratti da un dizionario:
{ground_truth_senses}

Lista di sensi estratti da un modello di linguaggio:
{predicted_senses}