# Usage:
# bash scripts/extract_contours.sh

source env/bin/activate

echo "Erk"
python -m src.extract.extract_contours erk
echo "Boehme"
python -m src.extract.extract_contours boehme 
echo "Creighton"
python -m src.extract.extract_contours creighton 

echo "Han"
python -m src.extract.extract_contours han
echo "Natmin"
python -m src.extract.extract_contours natmin
echo "Shanxi"
python -m src.extract.extract_contours shanxi

echo "Antiphons"
python -m src.extract.extract_contours liber-antiphons
echo "Responsories"
python -m src.extract.extract_contours liber-responsories
echo "Alleluias"
python -m src.extract.extract_contours liber-alleluias

echo "Cantus antiphons"
python -m src.extract.extract_contours cantus-antiphon
echo "Cantus responsories"
python -m src.extract.extract_contours cantus-responsory