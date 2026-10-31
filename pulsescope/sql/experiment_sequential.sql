-- Cumulative conversion at each interim look: users exposed by day k, converted by day k.
WITH looks AS (
    SELECT UNNEST(LIST_APPEND(RANGE({start} + {step} - 1, {end} - 1, {step}), {end} - 1)) AS look_day
), exposed AS (
    SELECT user_id, variant, first_exposure_ts, first_exposure_day
    FROM experiment_assignments
    WHERE experiment_id = '{experiment_id}' AND n_variants = 1
), conv AS (
    SELECT e.user_id, MIN(s.day) AS first_conv_day
    FROM exposed e JOIN sessions s
      ON s.user_id = e.user_id AND s.session_start >= e.first_exposure_ts AND s.purchased = 1
    GROUP BY e.user_id
)
SELECT l.look_day, e.variant,
       COUNT(*)                                                        AS users,
       SUM(CASE WHEN c.first_conv_day <= l.look_day THEN 1 ELSE 0 END) AS conversions
FROM looks l
JOIN exposed e ON e.first_exposure_day <= l.look_day
LEFT JOIN conv c USING (user_id)
GROUP BY 1, 2
ORDER BY 1, 2;
