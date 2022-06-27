source env/bin/activate

echo "Erk"
python generate_data.py erk
echo "Boehme"
python generate_data.py boehme 
echo "Creighton"
python generate_data.py creighton 
echo "Han"
python generate_data.py han
echo "Natmin"
python generate_data.py natmin
echo "Shanxi"
python generate_data.py shanxi
echo "Antiphons"
python generate_data.py liber-antiphons
echo "Responsories"
python generate_data.py liber-responsories
echo "Alleluias"
python generate_data.py liber-alleluias

# echo "Representations"
# python representations.py