The next event I want to automate is what is called “Desarme”. The source of the information we can get from the two cartera files, T0 and T-1. In each file, there is a field named Estrategia
de Cobertura (note the carriage return in the middle).
We need to identify the trades using Numero de Operacion where the value in the Estrategia de Cobertura field has changed from anything that is different to “NO” (in the T-1 Cartera) to “NO” (in the T0 Cartera). This is the indication of a desarme event. Trades that appear in T-1 cartera and then simply do not appear in T0 should be ignored for this purpose.
In the root of this directory, I have left the Excel sheet which contains the rules: “Matriz Contable Consolidada v1.xlsx”. Please explore and understand it using whatever you need, some Python script or something to open the Excel sheet. Here you can see a bunch of rules which have the Evento Contable of “DESARME”. These are the rules that we need to filter on to validate this event in the accounting interface file.
In the accounting interface file, the records we are interested in have the following glosas:
-	Desarme Swap UF-$ (This is a Swap Moneda)
-	Desarme Swap UF-USD (This is a Swap Moneda)
-	Desarme Swap USD-CLP (This is a Swap Moneda)
-	Desarme Swap CHF-CLF (This is a Swap Moneda)
-	Desarme Swap ICP-CLP (This is a Swap Cámara)
-	Desarme Swap ICP-CLF (This is a Swap Cámara)
-	Desarme Swap Tasa USD-USD (This is a Swap Tasas)
-	Desarme Swap Tasa CLP-CLP (This is a Swap Tasas)
-	Desarme Swap Tasa CLF-CLF (This is a Swap Tasas)
-	Desarme Swap Tasa EUR-EUR (This is a Swap Tasas)
The Desarme event actually has three sub-events. They all have the same Glosa, but bear in mind the following: 
1.	One thing that happens is effectively the equivalent of a Termino. So, we should apply the same kind of logic that we do for Termino (amortization amounts, by pata).
2.	The third type is essentially the same as the Curse. We should apply the same kind of logic that we have for the Curse (notional amounts, by pata).
3.	And then the third type is effectively the mark to market. Not the reversa mark to market but the mark to market.
So those are kind of the three sub-events that would all be included within the event called Desarme. Please bear this in mind because it's the logic that we need to replicate from our validators for Curse, Termino and Mark to Market. We need to apply some kind of logic in a Desarme validation function. 
Before you get on to building any code here, please have a good look at the current code to see if you can use it for inspiration. Let me know if you have any questions; it's important you have a full understanding before we get into actually building code. Finally, when we do build code, please keep a separate function for validating the desarmes because I don't want it to interfere with other functions that are currently working.

