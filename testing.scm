#!/usr/env/bin guile
!#

(use-modules (ice-9 binary-ports))
(use-modules (ice-9 textual-ports))
(use-modules (ice-9 pretty-print))
(use-modules (scheme base))
(load "utils.scm")

(load-mnist testing-phase)
(newline) (display "MNIST testing set loaded. The testing set contains 10,000 labeled handwritten digits, in the form of 28x28 grayscale images.") (newline)

;; FORWARD PASS.
(define (layer-forward input layer activation-function)
  (let* ((weights (car layer))
		 (biases (cadr layer))
		 (z (map + (matrix-*-vector weights input) biases)))
	(if (eq? activation-function ReLU)
	  (map activation-function z)
	  (activation-function z))))

(define (forward-pass layers input activation-function)
  (if (null? (cdr layers)) 
      (layer-forward input (car layers) softmax)
	  (forward-pass (cdr layers) 
                    (layer-forward input (car layers) activation-function) 
                    activation-function))) 
;; DONE FORWARD PASS.

(define (argmax lst)
  (define (argmax-helper lst max-idx max-val curr-idx)
    (if (null? lst)
        max-idx
        (if (> (car lst) max-val)
            (argmax-helper (cdr lst) curr-idx (car lst) (+ curr-idx 1))
            (argmax-helper (cdr lst) max-idx max-val (+ curr-idx 1)))))
  (argmax-helper lst 0 (car lst) 0))

(define (test-model layers num-test-samples)
  (let loop ((remaining-samples num-test-samples)
             (correct 0))
    (if (= remaining-samples 0)
        (begin
          (display "Accuracy: ")
          (display (* 100.0 (/ correct num-test-samples)))
          (display "%\n")
          (/ correct num-test-samples))
        (let* ((input (load-next-image))
               (true-label (read-label labels-port))
               (model-output (forward-pass layers input ReLU))
               (predicted-label (argmax model-output)))
          (if (= predicted-label true-label)
              (loop (- remaining-samples 1) (+ correct 1))
              (loop (- remaining-samples 1) correct))))))

(define trained-layers (load-model "trained-model"))
(test-model trained-layers 10000)
