#!/usr/env/bin guile
!#

(use-modules (ice-9 binary-ports))
(use-modules (ice-9 textual-ports))
(use-modules (ice-9 pretty-print))
(use-modules (scheme base))

;; LOAD MNIST DATASET.
(define (read-u32 port)
  (let ((b1 (read-u8 port))
        (b2 (read-u8 port))
        (b3 (read-u8 port))
        (b4 (read-u8 port)))
    (+ (ash b1 24) (ash b2 16) (ash b3 8) b4)))

(define training-phase 0)
(define testing-phase 1)

(define images-port #f)
(define labels-port #f)
(define images-magic #f)
(define num-images #f)
(define num-rows #f)
(define num-cols #f)
(define labels-magic #f)
(define num-labels #f)

(define (load-mnist phase)
  (set! images-port
        (open-input-file (if (= phase training-phase) "train-images-idx3-ubyte" "t10k-images-idx3-ubyte") #:binary #t))
  (set! labels-port
        (open-input-file (if (= phase training-phase) "train-labels-idx1-ubyte" "t10k-labels-idx1-ubyte") #:binary #t))
  (set! images-magic (read-u32 images-port))
  (set! num-images (read-u32 images-port))
  (set! num-rows (read-u32 images-port))
  (set! num-cols (read-u32 images-port))
  (set! labels-magic (read-u32 labels-port))
  (set! num-labels (read-u32 labels-port))
  (check-mnist phase))

(define (check-mnist phase)
  (cond ((not (= images-magic 2051)) 
		 (error "Magic number for images does not match the expected value (2051)"))
		((not (= labels-magic 2049))
		 (error "Magic number for labels does not match the expected value (2049)"))
		((not (= num-images (if (= phase training-phase) 60000 10000)))
		 (error (string-append 
			  "Number of images does not match the expected value for the MNIST " 
			  (if (= phase training-phase) "training" "testing") " set (" 
			  (if (= phase training-phase) "60000" "10000") ")")))
	 ((not (= num-labels (if (= phase training-phase) 60000 10000)))
     (error (string-append
			  "Number of labels does not match the expected value for the MNIST " 
			  (if (= phase training-phase) "training" "testing") " set (" 
			  (if (= phase training-phase) "60000" "10000") ")")))
	 ((and (not (= num-rows 28)) (not (= num-cols 28)))
	  (error "Image size does not match the expected value for the MNIST dataset (28x28)"))
	 (else #t)))

;(define (get-permuted-mnist-data phase)
;  (permute-list (get-all-mnist-data phase)))


(define (get-permuted-mnist-data phase)
  (permute-list (get-all-mnist-data phase)))

  (define (get-all-mnist-data phase)
  (load-mnist phase)
  (define (get-all-mnist-data-iter i data)
    (if (= i num-images)
        data
        (get-all-mnist-data-iter (+ i 1) (cons (list (load-next-image) (load-next-label)) data))))
  (let ((result (get-all-mnist-data-iter 0 '())))
	(close-mnist)
	result))

(define (remove-nth lst n)
  (if (= n 0)
	(cdr lst)
	(cons (car lst) (remove-nth (cdr lst) (- n 1)))))

(define (permute-list lst)
  (let* ((len (length lst))
		(strikes (fisher-yates-shuffle len))) 
	(define (permute-list-iter new-lst remaining-strikes n)
	  (if (= n 0) 
		new-lst
		(permute-list-iter (append new-lst (list (list-ref lst (car remaining-strikes)))) (cdr remaining-strikes) (- n 1))))
	(permute-list-iter '() strikes len)))

(define (fisher-yates-shuffle n)
  (define (fisher-yates-shuffle-iter original strikes)
    (if (= (length strikes) n)
	  strikes
        (let* ((k (random (length original))) 
			   (selected (list-ref original k))
			   (remaining (remove-nth original k)))
		  (fisher-yates-shuffle-iter remaining
									 (append strikes (list selected))))))
  (fisher-yates-shuffle-iter (iota n) '()))

(define (close-mnist)
  (close-input-port images-port)
  (close-input-port labels-port))

(define (read-image port)
  (get-bytevector-n port (* num-rows num-cols))) ; 28x28 = 784 bytes

(define (read-label port)
  (read-u8 port))

(define (normalize-image image-vector)
  (map (lambda (pixel) (/ pixel 255.0)) (u8vector->list image-vector)))

(define (one-hot-encode digit)
  (define (make-zero-list n) (make-list n 0))
  (let ((encoded (make-zero-list 10))) 
	(list-set! encoded digit 1) 
	encoded))

(define (load-next-image) 
  (normalize-image (read-image images-port)))

(define (load-next-label)
  (one-hot-encode (read-label labels-port)))
;; DONE LOAD MNIST DATASET.

;; MATRIX / VECTOR MANIPULATION. Most of the following matrix / vector manipulation code is from my solution to SICP Exercise 2.37.
(define (accumulate op initial sequence)
  (if (null? sequence)
	initial
	(op (car sequence)
		(accumulate op initial (cdr sequence)))))

(define (accumulate-n op init seqs)
  (if (null? (car seqs))
  '()
  (cons (accumulate op init (map car seqs))
        (accumulate-n op init (map cdr seqs)))))


(define (dot-product v w)
  (let ((products (map * v w)))
    (apply + products)))

(define (matrix-*-vector m v)
  (map (lambda (x) (dot-product x v)) m))

(define (transpose mat)
  (accumulate-n cons '() mat))

(define (matrix-*-matrix m n) 
  (let ((cols (transpose n)))
    (map (lambda (x) 
           (matrix-*-vector cols x)) 
         m)))

;; This prodedure is analogous to my solution to SICP exercise 2.31.
(define (tree-map proc . trees) ;; If passing in multiple trees, they should have the same structure.
  (apply map
         (lambda args
           (if (pair? (car args)) 
			 (apply tree-map proc args) 
			 (apply proc args))) 
		 trees))

(define (matrix-*-scalar m s)
  (tree-map (lambda (l) (* s l)) m))

(define (matrix-minus-matrix m1 m2)
  (tree-map (lambda (x y) (- x y)) m1 m2))

(define (outer-product v1 v2)
  (map (lambda (x) (map (lambda (y) (* x y)) v2)) v1))
;; DONE MATRIX / VECTOR MANIPULTAION.

;; ACTIVATION FUNCTIONS.
(define (ReLU x)
  (if (< x 0) 0 x))

(define (softmax logits)
  (let* ((exps (map exp logits))
         (sum (apply + exps)))
    (map (lambda (x) (/ x sum)) exps)))
;; DONE ACTIVATION FUNCTIONS.

(define (remove-last l)
  (reverse (cdr (reverse l))))

(define (last l)
  (car (reverse l)))

(define (save-model layers filename)
  (with-output-to-file filename
    (lambda ()
      (pretty-print layers))))

(define (load-model filename)
  (with-input-from-file filename
    (lambda ()
      (read))))


