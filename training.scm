#!/usr/env/bin guile
!#

(use-modules (ice-9 binary-ports))
(use-modules (scheme base))

(define (read-u32 port)
  (let ((b1 (read-u8 port))
        (b2 (read-u8 port))
        (b3 (read-u8 port))
        (b4 (read-u8 port)))
    (+ (ash b1 24) (ash b2 16) (ash b3 8) b4)))

;(define images-port (open-input-file "/home/jdafoe/MNIST/train-images-idx3-ubyte" #:binary #t))
;(define labels-port (open-input-file "/home/jdafoe/MNIST/train-labels-idx1-ubyte" #:binary #t))

;(define images-magic (read-u32 images-port))
;(define num-images (read-u32 images-port))
;(define num-rows (read-u32 images-port))
;(define num-cols (read-u32 images-port))

;(define labels-magic (read-u32 labels-port))
;(define num-labels (read-u32 labels-port))

;(display "Image magic: ") (display images-magic) (newline)
;(display "Labels magic: ") (display labels-magic) (newline)
;(display "Images: ") (display num-images) (newline)
;(display "Rows: ") (display num-rows) (newline)
;(display "Cols: ") (display num-cols) (newline)

;TODO: make sure biases work??

; The following matrix multiplication code is from my solution to SICP Exercise 2.37
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
  (accumulate + 0 (map * v w)))

(define (matrix-*-vector m v)
  (map (lambda (x) (dot-product x v)) m))

(define (transpose mat)
  (accumulate-n cons '() mat))

(define (matrix-*-matrix m n)
  (let ((cols (transpose n)))
    (map (lambda (x) 
           (matrix-*-vector cols x)) 
         m)))
; End matrix multiply

(define (ReLU x)
  (if (< x 0) 0 x))

(define (read-image port)
  (get-bytevector-n port (* num-rows num-cols))) ; 28x28 = 784 bytes

(define (read-label port)
  (read-u8 port))

(define (normalize-image image-vector)
  (map (lambda (pixel) (/ pixel 255.0)) (u8vector->list image-vector)))

(define (one-hot-encode label num-classes)
  (define (make-zero-vector n) (make-vector n 0))
  (define encoded-vector (make-zero-vector num-classes))
  (vector-set! encoded-vector label 1)
  encoded-vector)

(define (load-next-image) ; We need to include a 1 at the end for biases?
  (normalize-image (read-image images-port)))

(define (load-next-label)
  (one-hot-encode (read-label labels-port) 10))

(define (initialize-layer input-size output-size)
  (define (make-row n)
	(map (lambda (_) (+ 0 (* (sqrt (/ 2 input-size)))) (random:normal)) ; This is the He Initialization, suggested by ChatGPT.
		 (iota n)))
  (map (lambda (_) (make-row (+ 1 input-size))) (iota output-size))) ; The + 1 is for the bias. Should add an extra column.

(define (softmax logits)
  (let* ((exps (map exp logits))
         (sum (apply + exps)))
    (map (lambda (x) (/ x sum)) exps)))

; FORWARD PASS
(define (training-layer-forward input layer activation-function)
  (let ((zl (matrix-*-vector layer (cadr (car (reverse input))))))
	(append input (list (list (append zl '(1)) (append (map activation-function zl) '(1)))))))

; For the training forward-pass, we need to keep z^l for each layer.
(define (remove-last l)
  (reverse (cdr (reverse l))))

(define (identity x) x)
(define layer1 '((1 0)
                 (0 1)))
(define layer2 '((2 0)
				 (0 2)))
(define layers (list layer1 layer2))
(define input '((0 (1 2))))

(define (training-forward-pass layers input activation-function) ; AGHH this still needs work. it is complicated.
  (if (null? layers)
	(append (remove-last input) (append (list (list (car (car (reverse input))) (softmax (remove-last (car (car (reverse input))))))))) ; is the remove necessary? i think not. 1 is not added to zl.
	(training-forward-pass (cdr layers) 
				  (training-layer-forward input (car layers) activation-function)
				  activation-function)))

(display (training-forward-pass layers input identity))

; END FORWARD-PASS

; Below are the training specific procedures

(define (mean-squared-error predicted-pd actual-pd n) 
  (/ n (apply + (map (lambda (predicted actual)
					   (expt (- predicted actual) 2)) 
					 predicted-pd 
					 actual-pd))))

(define (cross-entropy-loss predicted-pd actual-pd n) ; For multiple samples, you sum the cross entropy loss???
  (- 0 (apply + (map (lambda (predicted actual)
					   (* actual (log predicted)))
					 predicted-pd
					 actual-pd))))

(define (compute-gradients layers input expected-output activation-function) 
  (let* ((model-output (training-forward-pass layers input activation-function))
		 (output-error (compute-output-error (car (reverse model-output)) expected-output activation-function))
		 (gradients (backpropagate-layers (reverse (layers)) (reverse (remove-last model-output)) output-error activation-function))) ; Do I need to remove last element from layers? idk
	gradients)) ; I think to fix the error noted below, do not remove-last from model output. We will have zi be one ahead of ai?!!!

(define (compute-output-error layer-output expected-output activation-function)
  (let ((zL (car layer-output))
        (aL (cadr layer-output)))
    (map (lambda (a y z) (* (- a y) (derivative activation-function z)))
         aL expected-output zL)))

(define (backpropagate-layers layers model-output output-error activation-function)
  (define (backpropagate layers model-output err)
    (if (null? model-output)
        '()
        (let* ((next-layer-output (car model-output))
               (next-layer (car layers))
               (zi (car next-layer-output))
               (ai (cadr next-layer-output))
               (delta (compute-delta next-layer err zi activation-function))
               (gradients (compute-gradients-for-layer next-layer-output delta)))
          (cons gradients (backpropagate (cdr layers) (cdr model-output) delta)))))
  (backpropagate layers model-output output-error))

(define (compute-delta layer err zi activation-function) ; Im not sure if this is using the previous activations like it should.
  (map (lambda (e z)
         (* (matrix-*-vector (transpose (layer-weights layer)) e)
              (derivative activation-function z)))
       err zi))

(define (compute-gradients-for-layer layer-output delta)
  (let ((activations (cadr layer-output)))
    (let ((grad-W (outer-product delta activations)))
	  grad-W)))

(define (derivative activation-function z)
  (cond ((eq? activation-function 'sigmoid) (* (sigmoid z) (- 1 (sigmoid z))))
        ((eq? activation-function 'ReLU) (if (> z 0) 1 0))))




; THE BELOW IS GENERATE BY CHATGPT. I shoudl work to understand this. The math is complex but understandable.
; Note there is no explicit bias so this maybe should be different.
;(define (compute-gradients layers input expected-output)
;  (define (layer-backprop input layer output expected-output)
;    (let* ((delta (sub expected-output output))  ; for MSE loss
;           (grad-W (map (lambda (x) (* delta x)) (transpose input)))  ; gradient of weights
;           (grad-b delta))  ; gradient of bias
;      (list grad-W grad-b)))
;
;  (define (backprop-layers layers input expected-output)
;    (if (null? layers)
;        '()
;        (let ((layer (car layers))
;              (rest (cdr layers)))
;          (let* ((output (layer-forward input layer ReLU))
;                 (gradients (layer-backprop input layer output expected-output)))
;            (cons gradients (backprop-layers rest output expected-output))))))
;
;  (backprop-layers layers input expected-output))
;
;
;(define (update-weights weights gradients learning-rate)
;  (map (lambda (w g) (map (lambda (wi gi) (- wi (* learning-rate gi))) w g)) weights gradients))
;
;(define (train-step layers input expected-output learning-rate)
;  (let* ((output (forward-pass layers input ReLU))
;         (loss (cross-entropy-loss output expected-output 1))  ; single sample
;         (gradients (compute-gradients layers input expected-output)))
;    (define new-weights (update-weights layers gradients learning-rate))
;    new-weights))
;
;
;(define (train-epoch layers num-steps learning-rate)
;  (define (train-step-loop remaining-steps)
;    (if (= remaining-steps 0)
;        '()  ; End of training step loop
;        (let* ((input (load-next-image))   ; Load one training image
;               (expected-output (load-next-label)) ; Load one-hot encoded label
;               (new-weights (train-step layers input expected-output learning-rate)))  ; Train on this sample
;          (train-step-loop (- remaining-steps 1))))) ; Recur for the next step
;  
;  (train-step-loop num-steps))
;
;
;(define (train-mnist layers epochs num-steps learning-rate)
;  (define (train-loop epoch)
;    (if (= epoch epochs)
;        '()  ; End of training loop
;        (begin
;          (display "Training epoch: ") (display epoch) (newline)
;          (train-epoch layers num-steps learning-rate)
;          (train-loop (+ epoch 1)))))
;
;  (train-loop 0))  ; Start training from epoch 0
;
;
;(define layers (list (initialize-layer 784 128) ; Example layers
;                     (initialize-layer 128 64)
;                     (initialize-layer 64 10)))
;(train-mnist layers 10 1000 0.01)  ; Train for 10 epochs, 1000 steps per epoch, learning rate 0.01
;
;

