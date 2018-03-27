#!/bin/bash
PlotText() {
   local _REG=$1
   local _text=$2
   local _XS=$3
   local _YS=$4

   #local _lb=('a' 'b' 'c' 'd' 'e' 'f')
   #local _title=`echo ${_text} | awk -v lb=${_lb[ifile]} '{print "("lb")  "$0}'`
   local _title=$_text
   #echo ${_title} 
   local _llon=`echo $_REG | sed s/'\-R'/''/ | awk -F/ -v xs=$_XS '{print $1+xs*($2-$1)}'`
   local _ulat=`echo $_REG | sed s/'\-R'/''/ | awk -F/ -v ys=$_YS '{print $4+ys*($4-$3)}'`
   #echo $_title | awk -v llon=$_llon -v ulat=$_ulat '{print llon, ulat, $0}'
   #echo $_title | awk -v llon=$_llon -v ulat=$_ulat 'BEGIN{OFS=","} {print llon, ulat, $0}' | pstext -R -J -F+a0+jLT+f12p -Glightgrey -W -TO -O -K -N >> $psout
   echo $_title | awk -v llon=$_llon -v ulat=$_ulat '{print llon, ulat, "12. 0. 20 LT", $0}' | pstext -R -J -Wwheat -O -K -N >> $psout
   #let ifile++
}

exeFilt() {
    sac <<EOF
    read $1
    bp c $3 $4 n $5 p 2
    write ${7}
    quit
EOF
}

### main ###
exePS=/home/tianye/usr/bin/pssac
# exeFilt=~/code/Programs/SACOperation/SAC_filter
exeFold=/work2/tianye/Thesis/Figures_OBS_Denoise/RecordSection/SAC_fold
exeReverse=/work2/tianye/Thesis/Figures_OBS_Denoise/RecordSection/SAC_reverse
SAClst=./JdF_BHZ.lst
sta=J73A

# gmt setttings
gmtset BASEMAP_TYPE plain
gmtset HEADER_FONT_SIZE 12
gmtset LABEL_FONT_SIZE 10
gmtset ANNOT_FONT_SIZE 8
gmtset HEADER_OFFSET 0.
gmtset LABEL_OFFSET -0.15
gmtset ANNOT_OFFSET 0.05

psout=RecordSection_${sta}_.ps
rm -f ${psout}
echo "0 0" | psxy -Rg -Jx1 -X-9 -Y12 -K -P > $psout
REG=-R-800/800/0/600
SCA=-JX8/-10

labels=('a' 'b'); ilabel=0
for ifreq in 1 2; do
    dirSACout=./SAC_filtered
    if [ $ifreq == 1 ]; then
        fl=0.05; fu=0.083333333333; title="12-20 sec"; norm1=20/0.5; norm2=30/0.5
    else
        fl=0.033; fu=0.05; title="20-30 sec"; norm1=60/0.5; norm2=80/0.5
    fi
    [[ -z `ls ${dirSACout}/COR_*${sta}*.SAC_ft${ifreq} 2>/dev/null` ]] && (
    mkdir -p ${dirSACout}
    for fsac in `cat ${SAClst} | grep _${sta}`; do
        sacname=$(basename $fsac)
        exeFilt ${fsac} -1 ${fl} ${fu} 6 5 ${dir} ${dirSACout}/${sacname}_ft${ifreq} 1>/dev/null
        $exeReverse ${dirSACout}/${sacname}_ft${ifreq} ${sta}
    done
    )
    psbasemap $REG $SCA -Ba400f100:"Lag Time(sec)":/a200/f50:"Distance (km)":WeSn -X9.8 -Y0 -O -K >> $psout
    $exePS -R -J -W0.5p,black -O -K ${dirSACout}/COR_*${sta}*.SAC_ft${ifreq} -Ekt -M50/0.5 >> $psout
    echo -e "0 0\n0 9999" | psxy -R -J -W3,red -O -K >> $psout
    PlotText $REG "(${labels[ilabel]})  ${title}" 0.005 -1.05; let ilabel++
done

echo "0 0" | psxy -R -J -O >> $psout
echo $psout
