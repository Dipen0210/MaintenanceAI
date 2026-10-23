ALTER TABLE model_activations ADD COLUMN plant_id TEXT REFERENCES plants(id);
UPDATE model_activations SET plant_id=(SELECT plant_id FROM users WHERE users.id=model_activations.actor_id);
CREATE TABLE plant_model_selections(plant_id TEXT NOT NULL REFERENCES plants(id), family TEXT NOT NULL, model_id TEXT NOT NULL REFERENCES model_versions(id), PRIMARY KEY(plant_id,family));
INSERT INTO plant_model_selections(plant_id,family,model_id) SELECT p.id,m.family,m.id FROM plants p CROSS JOIN model_versions m WHERE m.active=1;
