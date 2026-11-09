-- Q1 WHERE then HAVING (expect 30 customers)
SELECT customer_id, SUM(order_amount) AS total
FROM orders WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id HAVING SUM(order_amount) > 8000;

-- Q2 three-table join (expect 30 rows, total 2,998,668.80, max Bengaluru/Electronics 233,909.45)
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category ORDER BY c.city, p.category;

-- Q3 scalar subquery (expect 11)
SELECT * FROM products WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4 LEFT JOIN (expect 25)
SELECT c.customer_id
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL;

-- Q5 CTEs (expect 550 customers, average 2552.06)
WITH spend AS (
  SELECT customer_id, SUM(order_amount) AS total_spend
  FROM orders WHERE status = 'completed' GROUP BY customer_id
), avg_spend AS (SELECT AVG(total_spend) AS avg_total FROM spend)
SELECT s.customer_id, ROUND(s.total_spend, 2) AS total_spend, ROUND(a.avg_total, 2) AS avg_total
FROM spend s, avg_spend a WHERE s.total_spend > a.avg_total ORDER BY s.total_spend DESC;

-- Q6 ranking windows (expect 8 rows where rnk != row_num)
WITH spend AS (
  SELECT o.customer_id, c.city, SUM(o.order_amount) AS total_spend
  FROM orders o JOIN customers c ON o.customer_id = c.customer_id
  WHERE o.status = 'completed' GROUP BY o.customer_id, c.city
)
SELECT customer_id, city, ROUND(total_spend, 2) AS total_spend,
  RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
  DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
  ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend ORDER BY city, rnk;

-- Q7 running total (expect 183 rows, last cumulative 2,998,668.80)
WITH daily AS (
  SELECT substr(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
  FROM orders WHERE status = 'completed' GROUP BY day
)
SELECT day, ROUND(revenue, 2) AS revenue,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily ORDER BY day;