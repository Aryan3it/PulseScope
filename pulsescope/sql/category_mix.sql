-- Revenue share and month-over-month growth by category (window aggregates).
WITH m AS (
    SELECT DATE_TRUNC('month', event_date) AS month, category,
           SUM(revenue) AS revenue, COUNT(*) AS orders
    FROM orders GROUP BY 1, 2
)
SELECT month, category, revenue, orders,
       revenue / SUM(revenue) OVER (PARTITION BY month)                       AS revenue_share,
       revenue / LAG(revenue) OVER (PARTITION BY category ORDER BY month) - 1 AS mom_growth,
       RANK() OVER (PARTITION BY month ORDER BY revenue DESC)                 AS rank_in_month
FROM m
ORDER BY month, rank_in_month;
