set terminal pdf

set output "comparison.pdf"

set logscale x
set xrange[50:20000]
set yrange[10:103]
plot 'outputs/DemoProp_mic1_raw_spectrum.txt' u 1:6 w l lw 2 ti 'Pred', \
	'Meas/SPL_FO1_Pipistrel_Mic0_cut.dat' u 1:2 w l lw 2 ti 'Meas'
#	'Measurements/measMic1J0bp0deg30rps_Fig9b.txt' u 1:2 w l lw 2 ti 'Meas'
#	'SPL_rotor0/PipistrelProp_FO_SPLH_Mic0.txt' u 1:4 w p pt 7,\
