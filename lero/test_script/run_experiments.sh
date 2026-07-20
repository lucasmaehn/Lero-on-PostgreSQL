
SCALE=1
CONTAINER_NAME=lerodb
DATASET_DIR=./tpch_min.txt

docker compose -f ../../../docker-compose.yml up -d

for f in *.exp_conf; do
	echo "Running experiment: $f NO_ANALYZE"
	(cd ./../../../ && ./prepare-db.sh $CONTAINER_NAME ./Lero-on-PostgreSQL/lero/test_script/$f $SCALE)
	sleep 10
	python generate_dataset.py --dataset $DATASET_DIR --output ${f}__NO_ANALYZE

	(cd ./../../../ && ./prepare-db.sh $CONTAINER_NAME ./Lero-on-PostgreSQL/lero/test_script/$f $SCALE)
	sleep 10
	docker exec "${CONTAINER_NAME}" psql -U "postgres" -d "tpch_${SCALE}" -c "ANALYZE;"
	python generate_dataset.py --dataset $DATASET_DIR --output ${f}__WITH_ANALYZE
done


