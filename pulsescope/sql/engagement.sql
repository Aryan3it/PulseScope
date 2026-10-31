-- DAU / rolling WAU / rolling MAU and stickiness (DAU/MAU) using range-framed self joins.
WITH daily AS (
    SELECT DISTINCT event_date, user_id FROM sessions
), dates AS (
    SELECT DISTINCT event_date FROM daily
)
SELECT d.event_date,
       COUNT(DISTINCT CASE WHEN a.event_date = d.event_date THEN a.user_id END)                       AS dau,
       COUNT(DISTINCT CASE WHEN a.event_date > d.event_date - INTERVAL 7 DAY THEN a.user_id END)      AS wau,
       COUNT(DISTINCT a.user_id)                                                                       AS mau,
       COUNT(DISTINCT CASE WHEN a.event_date = d.event_date THEN a.user_id END)
           / COUNT(DISTINCT a.user_id)                                                                 AS stickiness
FROM dates d
JOIN daily a ON a.event_date > d.event_date - INTERVAL 30 DAY AND a.event_date <= d.event_date
WHERE d.event_date >= (SELECT MIN(event_date) FROM dates) + INTERVAL 29 DAY
GROUP BY d.event_date
ORDER BY d.event_date;
