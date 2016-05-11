#!/bin/bash

#Innanzitutto bisogna vedere se funziona cambiare i thread (farlo in c diretto??)
./bfs_omp -V 1 -g 1 -S 18 -E 150 > graph_1.txt

# Poi bisogna farlo per entrambi i tipi di grafo
#./bfs_omp -V 1 -g 1 -S 25 -E 25 >> graph_2.txt
