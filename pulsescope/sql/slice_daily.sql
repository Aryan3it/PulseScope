-- Daily funnel counts by one dimension; feeds the root-cause drill-down.
SELECT event_date, day, {dim} AS slice,
       COUNT(*)        AS sessions,
       SUM(carted)     AS carts,
       SUM(checkout)   AS checkouts,
       SUM(purchased)  AS orders,
       SUM(revenue)    AS revenue
FROM sessions
GROUP BY ALL
ORDER BY event_date, slice;
