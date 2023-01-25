(define (problem toy-delivery) (:domain toy-delivery)
  (:objects
    loc0 loc1 loc2 loc3
  )
  (:init 
	(at-robby loc0)
	(satisfied loc1)
  )
  (:goal (and (satisfied loc1) (satisfied loc2)))
)
