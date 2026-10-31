-- Weekly signup-cohort retention: share of each cohort active N weeks after its first session.
WITH first_seen AS (
    SELECT user_id, MIN(event_date) AS first_date FROM sessions GROUP BY user_id
), activity AS (
    SELECT DISTINCT s.user_id,
           DATE_TRUNC('week', f.first_date)                                  AS cohort_week,
           CAST(FLOOR(DATE_DIFF('day', f.first_date, s.event_date) / 7) AS INT) AS week_n
    FROM sessions s JOIN first_seen f USING (user_id)
), cohort_size AS (
    SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_users FROM activity WHERE week_n = 0 GROUP BY 1
)
SELECT a.cohort_week, c.cohort_users, a.week_n,
       COUNT(DISTINCT a.user_id)                      AS active_users,
       COUNT(DISTINCT a.user_id) / c.cohort_users     AS retention
FROM activity a JOIN cohort_size c USING (cohort_week)
WHERE a.week_n <= 8
GROUP BY 1, 2, 3
ORDER BY 1, 3;
