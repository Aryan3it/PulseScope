-- Classic RFM scoring with NTILE quintiles over the observation window.
WITH base AS (
    SELECT user_id, recency_days, orders AS frequency, revenue AS monetary
    FROM user_features WHERE orders > 0
), scored AS (
    SELECT *,
           6 - NTILE(5) OVER (ORDER BY recency_days)  AS r_score,
           NTILE(5) OVER (ORDER BY frequency)         AS f_score,
           NTILE(5) OVER (ORDER BY monetary)          AS m_score
    FROM base
)
SELECT *,
       CASE
           WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'champions'
           WHEN r_score >= 3 AND f_score >= 3                  THEN 'loyal'
           WHEN r_score >= 4 AND f_score <= 2                  THEN 'new_or_promising'
           WHEN r_score <= 2 AND f_score >= 3                  THEN 'at_risk'
           WHEN r_score <= 2 AND f_score <= 2                  THEN 'hibernating'
           ELSE 'needs_attention'
       END AS rfm_segment
FROM scored;
