EXP_DIR=slqa-final-sat-01
TRAIN=save/train

if [ ! -d tmp ] ; then mkdir tmp ; fi
rsync -avz 40.121.37.9:cs224n-squad/$TRAIN/$EXP_DIR tmp/
if [ -f tmp/$EXP_DIR/best.pth.tar ] ; then
	rm -rf $TRAIN/$EXP_DIR
	mv tmp/$EXP_DIR $TRAIN
fi
