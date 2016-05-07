#!/usr/bin/gnuplot
set terminal png
set output "graf_floyd_intel.png"
set title "Graf výpočtu Floyd Warshall algoritmus -- Intel Xeon"
set ylabel "Čas"
set xlabel "Počet procesů"
plot "floyd_xeon_data.dat" using 1:((155.992)/$2) with lines title "Rychlost zpracování", \
x with lines title "Lineární zrychlení"

set terminal png
set output "graf_dijkstra_intel.png"
set title "Graf výpočtu Dijkstra algoritmu -- Intel Xeon"
set ylabel "Čas"
set xlabel "Počet vláken"
plot "dijskra_xeon_data.dat" using 1:((360.883)/$2) with lines title "Rychlost zpracování", \
x with lines title "Lineární zrychlení"

set terminal png
set output "graf_dijkstra_phi.png"
set title "Graf výpočtu Dijkstra algoritmu -- Xeon Phi"
set ylabel "Čas"
set xlabel "Počet vláken"
plot "dijskra_phi_data.dat" using 1:((2622.89)/$2) with lines title "Rychlost zpracování", \
x with lines title "Lineární zrychlení"

set terminal png
set output "graf_floyd_phi.png"
set title "Graf výpočtu Floyd Warshall algoritmu -- Xeon Phi"
set ylabel "Čas"
set xlabel "Počet vláken"
plot "floyd_phi_data.dat" using 1:(1337.32)/$2 with lines title "Rychlost zpracování", \
x with lines title "Lineární zrychlení"



set terminal png
set output "graf_floyd_gts_780_Ti.png"
set title "Graf výpočtu Floyd Warshall algoritmu -- NVidia GTS 780 Ti"
set ylabel "Čas"
set xlabel "Počet vláken"
plot "gen_gts_780_Ti.dat" using 1:(336745)/$2 with lines title "Rychlost zpracování", \
x with lines title "Lineární zrychlení"


set terminal png
set output "graf_floyd_gts_650_Ti.png"
set title "Graf výpočtu Floyd Warshall algoritmu -- NVidia GTS 650 Ti"
set ylabel "Čas"
set xlabel "Počet vláken"
plot "gen_gts_650_Ti.dat" using 1:(1006140)/$2 with lines title "Rychlost zpracování", \
x with lines title "Lineární zrychlení"
