#Scopo:

Sviluppo di un codice OpenMP e di un codice CUDA per la Breadth First Search su grafi costruiti con il metodo RMAT

#Caratteristiche:

* I codici dovranno essere integrati all'interno del driver reso disponibile (grazie ad Enrico Mastrostefano).
* Il driver può essere compilato semplicemente con il comando cc -o driverBFS driverBFS.c ma i codice completi CUDA ed OpenMP è opportuno che siano compilabili tramite makefile.
* Interessante cercare di velocizzare (eseguendolo in parallalelo) anche l'algoritmo di generazione del grafo
* Non è richiesto un codice OpenMP e CUDA ma due implementazioni distinte che utilizzano lo stesso driver
