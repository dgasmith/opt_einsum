Large Expressions with Greedy
-----------------------------

Using the greedy method allows the contraction of hundreds of tensors. Here's
an example from quantum of computing the inner product between two `'Matrix
Product States' <https://en.wikipedia.org/wiki/Matrix_product_state>`_.
Graphically, if we represent each tensor as an ``O``, give it
the same number of 'legs' as it has indices, and join those legs when that
index is summed with another tensor, we get an expression for ``n`` particles
that looks like::

    O-O-O-O-O-O-     -O-O-O-O-O-O
    | | | | | |  ...  | | | | | |
    O-O-O-O-O-O-     -O-O-O-O-O-O

    0 1 2 3 4 5 ........... n-2 n-1

The meaning of this is not that important other than its a large, useful
contraction. For ``n=100`` it involves 200 different tensors and about 300
unique indices. With this many indices it can be useful to generate them with
the function :func:`~opt_einsum.parser.get_symbol`.

Let's set up the required einsum string:

.. code-block:: python

    >>> import numpy as np
    >>> import opt_einsum as oe

    >>> n = 100
    >>> phys_dim = 3
    >>> bond_dim = 10

    >>> # start with first site
    ... # O--
    ... # |
    ... # O--
    >>> einsum_str = "ab,ac,"

    >>> for i in range(1, n - 1):
    ...     # set the upper left/right, middle and lower left/right indices
    ...     # --O--
    ...     #   |
    ...     # --O--
    ...     j = 3 * i
    ...     ul, ur, m, ll, lr = (oe.get_symbol(i)
    ...                          for i in (j - 1, j + 2, j, j - 2, j + 1))
    >>>     einsum_str += "{}{}{},{}{}{},".format(m, ul, ur, m, ll, lr)

    >>> # finish with last site
    ... # --O
    ... #   |
    ... # --O
    >>> i = n - 1
    >>> j = 3 * i
    >>> ul, m, ll, =  (oe.get_symbol(i) for i in (j - 1, j, j - 2))
    >>> einsum_str += "{}{},{}{}".format(m, ul, m, ll)

Generate the shapes:

.. code-block:: python

    >>> def gen_shapes():
    ...     yield (phys_dim, bond_dim)
    ...     yield (phys_dim, bond_dim)
    ...     for i in range(1, n - 1):
    ...         yield(phys_dim, bond_dim, bond_dim)
    ...         yield(phys_dim, bond_dim, bond_dim)
    ...     yield (phys_dim, bond_dim)
    ...     yield (phys_dim, bond_dim)

    >>> shapes = tuple(gen_shapes())


Let's time how long it takes to generate the expression (``'greedy'`` is used
by default, and we turn off the ``memory_limit``)::

    %timeit expr = oe.contract_expression(einsum_str, *shapes, memory_limit=-1)
    76.2 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

This is pretty manageable, though we might want to think about splitting the
expression up if we go a lot bigger.
Importantly, we can then use this repeatedly with any set of matching arrays:

.. code-block:: python

    >>> arrays = [np.random.randn(*shp) / 4 for shp in shapes]
    >>> expr(*arrays)
    array(23.23628116)

    >>> arrays = [np.random.randn(*shp) / 4 for shp in shapes]
    >>> expr(*arrays)
    array(-12.21091879)

And if we **really** want we can generate the full contraction path info:

.. code-block:: python

    >>> print(oe.contract_path(einsum_str, *arrays, memory_limit=-1)[1])
      Complete contraction:  ab,ac,dcf,dbe,gfi,geh,jil,jhk,mlo,mkn,por,pnq,sru,sqt,vux,vtw,yxA,ywz,BAD,BzC,EDG,ECF,HGJ,HFI,KJM,KIL,NMP,NLO,QPS,QOR,TSV,TRU,WVY,WUX,ZYÂ,ZXÁ,ÃÂÅ,ÃÁÄ,ÆÅÈ,ÆÄÇ,ÉÈË,ÉÇÊ,ÌËÎ,ÌÊÍ,ÏÎÑ,ÏÍÐ,ÒÑÔ,ÒÐÓ,ÕÔ×,ÕÓÖ,Ø×Ú,ØÖÙ,ÛÚÝ,ÛÙÜ,ÞÝà,ÞÜß,áàã,áßâ,äãæ,äâå,çæé,çåè,êéì,êèë,íìï,íëî,ðïò,ðîñ,óòõ,óñô,öõø,öô÷,ùøû,ù÷ú,üûþ,üúý,ÿþā,ÿýĀ,ĂāĄ,ĂĀă,ąĄć,ąăĆ,ĈćĊ,ĈĆĉ,ċĊč,ċĉČ,ĎčĐ,ĎČď,đĐē,đďĒ,ĔēĖ,ĔĒĕ,ėĖę,ėĕĘ,ĚęĜ,ĚĘě,ĝĜğ,ĝěĞ,ĠğĢ,ĠĞġ,ģĢĥ,ģġĤ,ĦĥĨ,ĦĤħ,ĩĨī,ĩħĪ,ĬīĮ,ĬĪĭ,įĮı,įĭİ,ĲıĴ,Ĳİĳ,ĵĴķ,ĵĳĶ,ĸķĺ,ĸĶĹ,ĻĺĽ,ĻĹļ,ľĽŀ,ľļĿ,ŁŀŃ,ŁĿł,ńŃņ,ńłŅ,Ňņŉ,ŇŅň,ŊŉŌ,Ŋňŋ,ōŌŏ,ōŋŎ,ŐŏŒ,ŐŎő,œŒŕ,œőŔ,ŖŕŘ,ŖŔŗ,řŘś,řŗŚ,ŜśŞ,ŜŚŝ,şŞš,şŝŠ,ŢšŤ,ŢŠţ,ťŤŧ,ťţŦ,ŨŧŪ,ŨŦũ,ūŪŭ,ūũŬ,ŮŭŰ,ŮŬů,űŰų,űůŲ,ŴųŶ,ŴŲŵ,ŷŶŹ,ŷŵŸ,źŹż,źŸŻ,Žżſ,ŽŻž,ƀſƂ,ƀžƁ,ƃƂƅ,ƃƁƄ,Ɔƅƈ,ƆƄƇ,ƉƈƋ,ƉƇƊ,ƌƋƎ,ƌƊƍ,ƏƎƑ,ƏƍƐ,ƒƑƔ,ƒƐƓ,ƕƔƗ,ƕƓƖ,ƘƗƚ,ƘƖƙ,ƛƚƝ,ƛƙƜ,ƞƝƠ,ƞƜƟ,ơƠƣ,ơƟƢ,ƤƣƦ,ƤƢƥ,ƧƦƩ,Ƨƥƨ,ƪƩƬ,ƪƨƫ,ƭƬƯ,ƭƫƮ,ưƯƲ,ưƮƱ,ƳƲƵ,ƳƱƴ,ƶƵ,ƶƴ->
             Naive scaling:  298
         Optimized scaling:  5
          Naive FLOP count:  1.031e+248
      Optimized FLOP count:  1.168e+06
       Theoretical speedup:  88264689284468460017580864156865782413140936705854966013600065426858041248009637246968036807489558012989638169986640870276510490846199301907401763236976204166215471281505344088317454144870323271826022036197984172898402324699098341524952317952.000
      Largest intermediate:  3.000e+02 elements
    --------------------------------------------------------------------------------
    scaling        BLAS                current                             remaining
    --------------------------------------------------------------------------------
       4           TDOT            dbe,ab->ade ac,dcf,gfi,geh,jil,jhk,mlo,mkn,por,pnq,sru,sqt,vux,vtw,yxA,ywz,BAD,BzC,EDG,ECF,HGJ,HFI,KJM,KIL,NMP,NLO,QPS,QOR,TSV,TRU,WVY,WUX,ZYÂ,ZXÁ,ÃÂÅ,ÃÁÄ,ÆÅÈ,ÆÄÇ,ÉÈË,ÉÇÊ,ÌËÎ,ÌÊÍ,ÏÎÑ,ÏÍÐ,ÒÑÔ,ÒÐÓ,ÕÔ×,ÕÓÖ,Ø×Ú,ØÖÙ,ÛÚÝ,ÛÙÜ,ÞÝà,ÞÜß,áàã,áßâ,äãæ,äâå,çæé,çåè,êéì,êèë,íìï,íëî,ðïò,ðîñ,óòõ,óñô,öõø,öô÷,ùøû,ù÷ú,üûþ,üúý,ÿþā,ÿýĀ,ĂāĄ,ĂĀă,ąĄć,ąăĆ,ĈćĊ,ĈĆĉ,ċĊč,ċĉČ,ĎčĐ,ĎČď,đĐē,đďĒ,ĔēĖ,ĔĒĕ,ėĖę,ėĕĘ,ĚęĜ,ĚĘě,ĝĜğ,ĝěĞ,ĠğĢ,ĠĞġ,ģĢĥ,ģġĤ,ĦĥĨ,ĦĤħ,ĩĨī,ĩħĪ,ĬīĮ,ĬĪĭ,įĮı,įĭİ,ĲıĴ,Ĳİĳ,ĵĴķ,ĵĳĶ,ĸķĺ,ĸĶĹ,ĻĺĽ,ĻĹļ,ľĽŀ,ľļĿ,ŁŀŃ,ŁĿł,ńŃņ,ńłŅ,Ňņŉ,ŇŅň,ŊŉŌ,Ŋňŋ,ōŌŏ,ōŋŎ,ŐŏŒ,ŐŎő,œŒŕ,œőŔ,ŖŕŘ,ŖŔŗ,řŘś,řŗŚ,ŜśŞ,ŜŚŝ,şŞš,şŝŠ,ŢšŤ,ŢŠţ,ťŤŧ,ťţŦ,ŨŧŪ,ŨŦũ,ūŪŭ,ūũŬ,ŮŭŰ,ŮŬů,űŰų,űůŲ,ŴųŶ,ŴŲŵ,ŷŶŹ,ŷŵŸ,źŹż,źŸŻ,Žżſ,ŽŻž,ƀſƂ,ƀžƁ,ƃƂƅ,ƃƁƄ,Ɔƅƈ,ƆƄƇ,ƉƈƋ,ƉƇƊ,ƌƋƎ,ƌƊƍ,ƏƎƑ,ƏƍƐ,ƒƑƔ,ƒƐƓ,ƕƔƗ,ƕƓƖ,ƘƗƚ,ƘƖƙ,ƛƚƝ,ƛƙƜ,ƞƝƠ,ƞƜƟ,ơƠƣ,ơƟƢ,ƤƣƦ,ƤƢƥ,ƧƦƩ,Ƨƥƨ,ƪƩƬ,ƪƨƫ,ƭƬƯ,ƭƫƮ,ưƯƲ,ưƮƱ,ƳƲƵ,ƳƱƴ,ƶƵ,ƶƴ,ade->
       4           TDOT            dcf,ac->adf gfi,geh,jil,jhk,mlo,mkn,por,pnq,sru,sqt,vux,vtw,yxA,ywz,BAD,BzC,EDG,ECF,HGJ,HFI,KJM,KIL,NMP,NLO,QPS,QOR,TSV,TRU,WVY,WUX,ZYÂ,ZXÁ,ÃÂÅ,ÃÁÄ,ÆÅÈ,ÆÄÇ,ÉÈË,ÉÇÊ,ÌËÎ,ÌÊÍ,ÏÎÑ,ÏÍÐ,ÒÑÔ,ÒÐÓ,ÕÔ×,ÕÓÖ,Ø×Ú,ØÖÙ,ÛÚÝ,ÛÙÜ,ÞÝà,ÞÜß,áàã,áßâ,äãæ,äâå,çæé,çåè,êéì,êèë,íìï,íëî,ðïò,ðîñ,óòõ,óñô,öõø,öô÷,ùøû,ù÷ú,üûþ,üúý,ÿþā,ÿýĀ,ĂāĄ,ĂĀă,ąĄć,ąăĆ,ĈćĊ,ĈĆĉ,ċĊč,ċĉČ,ĎčĐ,ĎČď,đĐē,đďĒ,ĔēĖ,ĔĒĕ,ėĖę,ėĕĘ,ĚęĜ,ĚĘě,ĝĜğ,ĝěĞ,ĠğĢ,ĠĞġ,ģĢĥ,ģġĤ,ĦĥĨ,ĦĤħ,ĩĨī,ĩħĪ,ĬīĮ,ĬĪĭ,įĮı,įĭİ,ĲıĴ,Ĳİĳ,ĵĴķ,ĵĳĶ,ĸķĺ,ĸĶĹ,ĻĺĽ,ĻĹļ,ľĽŀ,ľļĿ,ŁŀŃ,ŁĿł,ńŃņ,ńłŅ,Ňņŉ,ŇŅň,ŊŉŌ,Ŋňŋ,ōŌŏ,ōŋŎ,ŐŏŒ,ŐŎő,œŒŕ,œőŔ,ŖŕŘ,ŖŔŗ,řŘś,řŗŚ,ŜśŞ,ŜŚŝ,şŞš,şŝŠ,ŢšŤ,ŢŠţ,ťŤŧ,ťţŦ,ŨŧŪ,ŨŦũ,ūŪŭ,ūũŬ,ŮŭŰ,ŮŬů,űŰų,űůŲ,ŴųŶ,ŴŲŵ,ŷŶŹ,ŷŵŸ,źŹż,źŸŻ,Žżſ,ŽŻž,ƀſƂ,ƀžƁ,ƃƂƅ,ƃƁƄ,Ɔƅƈ,ƆƄƇ,ƉƈƋ,ƉƇƊ,ƌƋƎ,ƌƊƍ,ƏƎƑ,ƏƍƐ,ƒƑƔ,ƒƐƓ,ƕƔƗ,ƕƓƖ,ƘƗƚ,ƘƖƙ,ƛƚƝ,ƛƙƜ,ƞƝƠ,ƞƜƟ,ơƠƣ,ơƟƢ,ƤƣƦ,ƤƢƥ,ƧƦƩ,Ƨƥƨ,ƪƩƬ,ƪƨƫ,ƭƬƯ,ƭƫƮ,ưƯƲ,ưƮƱ,ƳƲƵ,ƳƱƴ,ƶƵ,ƶƴ,ade,adf->
       4           GEMM            ƶƵ,ƳƲƵ->ƳƶƲ gfi,geh,jil,jhk,mlo,mkn,por,pnq,sru,sqt,vux,vtw,yxA,ywz,BAD,BzC,EDG,ECF,HGJ,HFI,KJM,KIL,NMP,NLO,QPS,QOR,TSV,TRU,WVY,WUX,ZYÂ,ZXÁ,ÃÂÅ,ÃÁÄ,ÆÅÈ,ÆÄÇ,ÉÈË,ÉÇÊ,ÌËÎ,ÌÊÍ,ÏÎÑ,ÏÍÐ,ÒÑÔ,ÒÐÓ,ÕÔ×,ÕÓÖ,Ø×Ú,ØÖÙ,ÛÚÝ,ÛÙÜ,ÞÝà,ÞÜß,áàã,áßâ,äãæ,äâå,çæé,çåè,êéì,êèë,íìï,íëî,ðïò,ðîñ,óòõ,óñô,öõø,öô÷,ùøû,ù÷ú,üûþ,üúý,ÿþā,ÿýĀ,ĂāĄ,ĂĀă,ąĄć,ąăĆ,ĈćĊ,ĈĆĉ,ċĊč,ċĉČ,ĎčĐ,ĎČď,đĐē,đďĒ,ĔēĖ,ĔĒĕ,ėĖę,ėĕĘ,ĚęĜ,ĚĘě,ĝĜğ,ĝěĞ,ĠğĢ,ĠĞġ,ģĢĥ,ģġĤ,ĦĥĨ,ĦĤħ,ĩĨī,ĩħĪ,ĬīĮ,ĬĪĭ,įĮı,įĭİ,ĲıĴ,Ĳİĳ,ĵĴķ,ĵĳĶ,ĸķĺ,ĸĶĹ,ĻĺĽ,ĻĹļ,ľĽŀ,ľļĿ,ŁŀŃ,ŁĿł,ńŃņ,ńłŅ,Ňņŉ,ŇŅň,ŊŉŌ,Ŋňŋ,ōŌŏ,ōŋŎ,ŐŏŒ,ŐŎő,œŒŕ,œőŔ,ŖŕŘ,ŖŔŗ,řŘś,řŗŚ,ŜśŞ,ŜŚŝ,şŞš,şŝŠ,ŢšŤ,ŢŠţ,ťŤŧ,ťţŦ,ŨŧŪ,ŨŦũ,ūŪŭ,ūũŬ,ŮŭŰ,ŮŬů,űŰų,űůŲ,ŴųŶ,ŴŲŵ,ŷŶŹ,ŷŵŸ,źŹż,źŸŻ,Žżſ,ŽŻž,ƀſƂ,ƀžƁ,ƃƂƅ,ƃƁƄ,Ɔƅƈ,ƆƄƇ,ƉƈƋ,ƉƇƊ,ƌƋƎ,ƌƊƍ,ƏƎƑ,ƏƍƐ,ƒƑƔ,ƒƐƓ,ƕƔƗ,ƕƓƖ,ƘƗƚ,ƘƖƙ,ƛƚƝ,ƛƙƜ,ƞƝƠ,ƞƜƟ,ơƠƣ,ơƟƢ,ƤƣƦ,ƤƢƥ,ƧƦƩ,Ƨƥƨ,ƪƩƬ,ƪƨƫ,ƭƬƯ,ƭƫƮ,ưƯƲ,ưƮƱ,ƳƱƴ,ƶƴ,ade,adf,ƳƶƲ->
       4           GEMM            ƶƴ,ƳƱƴ->ƳƶƱ gfi,geh,jil,jhk,mlo,mkn,por,pnq,sru,sqt,vux,vtw,yxA,ywz,BAD,BzC,EDG,ECF,HGJ,HFI,KJM,KIL,NMP,NLO,QPS,QOR,TSV,TRU,WVY,WUX,ZYÂ,ZXÁ,ÃÂÅ,ÃÁÄ,ÆÅÈ,ÆÄÇ,ÉÈË,ÉÇÊ,ÌËÎ,ÌÊÍ,ÏÎÑ,ÏÍÐ,ÒÑÔ,ÒÐÓ,ÕÔ×,ÕÓÖ,Ø×Ú,ØÖÙ,ÛÚÝ,ÛÙÜ,ÞÝà,ÞÜß,áàã,áßâ,äãæ,äâå,çæé,çåè,êéì,êèë,íìï,íëî,ðïò,ðîñ,óòõ,óñô,öõø,öô÷,ùøû,ù÷ú,üûþ,üúý,ÿþā,ÿýĀ,ĂāĄ,ĂĀă,ąĄć,ąăĆ,ĈćĊ,ĈĆĉ,ċĊč,ċĉČ,ĎčĐ,ĎČď,đĐē,đďĒ,ĔēĖ,ĔĒĕ,ėĖę,ėĕĘ,ĚęĜ,ĚĘě,ĝĜğ,ĝěĞ,ĠğĢ,ĠĞġ,ģĢĥ,ģġĤ,ĦĥĨ,ĦĤħ,ĩĨī,ĩħĪ,ĬīĮ,ĬĪĭ,įĮı,įĭİ,ĲıĴ,Ĳİĳ,ĵĴķ,ĵĳĶ,ĸķĺ,ĸĶĹ,ĻĺĽ,ĻĹļ,ľĽŀ,ľļĿ,ŁŀŃ,ŁĿł,ńŃņ,ńłŅ,Ňņŉ,ŇŅň,ŊŉŌ,Ŋňŋ,ōŌŏ,ōŋŎ,ŐŏŒ,ŐŎő,œŒŕ,œőŔ,ŖŕŘ,ŖŔŗ,řŘś,řŗŚ,ŜśŞ,ŜŚŝ,şŞš,şŝŠ,ŢšŤ,ŢŠţ,ťŤŧ,ťţŦ,ŨŧŪ,ŨŦũ,ūŪŭ,ūũŬ,ŮŭŰ,ŮŬů,űŰų,űůŲ,ŴųŶ,ŴŲŵ,ŷŶŹ,ŷŵŸ,źŹż,źŸŻ,Žżſ,ŽŻž,ƀſƂ,ƀžƁ,ƃƂƅ,ƃƁƄ,Ɔƅƈ,ƆƄƇ,ƉƈƋ,ƉƇƊ,ƌƋƎ,ƌƊƍ,ƏƎƑ,ƏƍƐ,ƒƑƔ,ƒƐƓ,ƕƔƗ,ƕƓƖ,ƘƗƚ,ƘƖƙ,ƛƚƝ,ƛƙƜ,ƞƝƠ,ƞƜƟ,ơƠƣ,ơƟƢ,ƤƣƦ,ƤƢƥ,ƧƦƩ,Ƨƥƨ,ƪƩƬ,ƪƨƫ,ƭƬƯ,ƭƫƮ,ưƯƲ,ưƮƱ,ade,adf,ƳƶƲ,ƳƶƱ->
       5           TDOT          ade,geh->adgh gfi,jil,jhk,mlo,mkn,por,pnq,sru,sqt,vux,vtw,yxA,ywz,BAD,BzC,EDG,ECF,HGJ,HFI,KJM,KIL,NMP,NLO,QPS,QOR,TSV,TRU,WVY,WUX,ZYÂ,ZXÁ,ÃÂÅ,ÃÁÄ,ÆÅÈ,ÆÄÇ,ÉÈË,ÉÇÊ,ÌËÎ,ÌÊÍ,ÏÎÑ,ÏÍÐ,ÒÑÔ,ÒÐÓ,ÕÔ×,ÕÓÖ,Ø×Ú,ØÖÙ,ÛÚÝ,ÛÙÜ,ÞÝà,ÞÜß,áàã,áßâ,äãæ,äâå,çæé,çåè,êéì,êèë,íìï,íëî,ðïò,ðîñ,óòõ,óñô,öõø,öô÷,ùøû,ù÷ú,üûþ,üúý,ÿþā,ÿýĀ,ĂāĄ,ĂĀă,ąĄć,ąăĆ,ĈćĊ,ĈĆĉ,ċĊč,ċĉČ,ĎčĐ,ĎČď,đĐē,đďĒ,ĔēĖ,ĔĒĕ,ėĖę,ėĕĘ,ĚęĜ,ĚĘě,ĝĜğ,ĝěĞ,ĠğĢ,ĠĞġ,ģĢĥ,ģġĤ,ĦĥĨ,ĦĤħ,ĩĨī,ĩħĪ,ĬīĮ,ĬĪĭ,įĮı,įĭİ,ĲıĴ,Ĳİĳ,ĵĴķ,ĵĳĶ,ĸķĺ,ĸĶĹ,ĻĺĽ,ĻĹļ,ľĽŀ,ľļĿ,ŁŀŃ,ŁĿł,ńŃņ,ńłŅ,Ňņŉ,ŇŅň,ŊŉŌ,Ŋňŋ,ōŌŏ,ōŋŎ,ŐŏŒ,ŐŎő,œŒŕ,œőŔ,ŖŕŘ,ŖŔŗ,řŘś,řŗŚ,ŜśŞ,ŜŚŝ,şŞš,şŝŠ,ŢšŤ,ŢŠţ,ťŤŧ,ťţŦ,ŨŧŪ,ŨŦũ,ūŪŭ,ūũŬ,ŮŭŰ,ŮŬů,űŰų,űůŲ,ŴųŶ,ŴŲŵ,ŷŶŹ,ŷŵŸ,źŹż,źŸŻ,Žżſ,ŽŻž,ƀſƂ,ƀžƁ,ƃƂƅ,ƃƁƄ,Ɔƅƈ,ƆƄƇ,ƉƈƋ,ƉƇƊ,ƌƋƎ,ƌƊƍ,ƏƎƑ,ƏƍƐ,ƒƑƔ,ƒƐƓ,ƕƔƗ,ƕƓƖ,ƘƗƚ,ƘƖƙ,ƛƚƝ,ƛƙƜ,ƞƝƠ,ƞƜƟ,ơƠƣ,ơƟƢ,ƤƣƦ,ƤƢƥ,ƧƦƩ,Ƨƥƨ,ƪƩƬ,ƪƨƫ,ƭƬƯ,ƭƫƮ,ưƯƲ,ưƮƱ,adf,ƳƶƲ,ƳƶƱ,adgh->

       ...

       4           TDOT            Ğğ,ĠğĢ->ĠĞĢ                  ĠĞġ,ģĢĥ,ģġĤ,Ĥĥ,ĠĞĢ->
       4           GEMM            ĠĞĢ,ĠĞġ->ġĢ                       ģĢĥ,ģġĤ,Ĥĥ,ġĢ->
       4           GEMM            Ĥĥ,ģĢĥ->ģĢĤ                          ģġĤ,ġĢ,ģĢĤ->
       4           TDOT            ģĢĤ,ģġĤ->ġĢ                               ġĢ,ġĢ->
       2            DOT                ġĢ,ġĢ->                                    ->

Where we can see the speedup over a naive einsum is about ``10^241``, not bad!
