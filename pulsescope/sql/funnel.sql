-- Session funnel by platform with step-to-step conversion (window functions over a long-format funnel).
WITH steps AS (
    SELECT platform, 1 AS step, 'session'  AS stage, COUNT(*)       AS n FROM sessions GROUP BY platform
    UNION ALL SELECT platform, 2, 'cart',     SUM(carted)    FROM sessions GROUP BY platform
    UNION ALL SELECT platform, 3, 'checkout', SUM(checkout)  FROM sessions GROUP BY platform
    UNION ALL SELECT platform, 4, 'purchase', SUM(purchased) FROM sessions GROUP BY platform
)
SELECT platform, step, stage, n,
       n / LAG(n) OVER (PARTITION BY platform ORDER BY step)          AS step_conversion,
       n / FIRST_VALUE(n) OVER (PARTITION BY platform ORDER BY step)  AS cumulative_conversion
FROM steps
ORDER BY platform, step;
