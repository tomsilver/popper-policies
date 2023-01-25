(define (problem toy-delivery) (:domain toy-delivery)
  (:objects
    loc4 loc5 loc6 loc7 loc8
  )
  (:init 
	(at-robby loc5)
  (satisfied loc6)
  (satisfied loc7)
  )
  (:goal (and (satisfied loc6) (satisfied loc7) (satisfied loc8)))
)
