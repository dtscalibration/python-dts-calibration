2. Common DataStore functions
=============================

Examples of how to do some of the more commonly used functions:

1. mean, min, max, std
2. Selecting
3. Selecting by index
4. Downsample (time dimension)
5. Upsample / Interpolation (length and time dimension)

.. code:: ipython3

    import os
    
    from dtscalibration import read_silixa_files


.. parsed-literal::

    /Users/bfdestombe/anaconda3/envs/dts/lib/python3.7/typing.py:847: FutureWarning: xarray subclass DataStore should explicitly define __slots__
      super().__init_subclass__(*args, **kwargs)


First we load the raw measurements into a ``DataStore`` object, as we
learned from the previous notebook.

.. code:: ipython3

    filepath = os.path.join('..', '..', 'tests', 'data', 'single_ended')
    
    ds = read_silixa_files(
        directory=filepath,
        timezone_netcdf='UTC',
        file_ext='*.xml')


.. parsed-literal::

    3 files were found, each representing a single timestep
    4 recorded vars were found: LAF, ST, AST, TMP
    Recorded at 1461 points along the cable
    The measurement is single ended
    Reading the data from disk


0 Access the data
-----------------

The implemented read routines try to read as much data from the raw DTS
files as possible. Usually they would have coordinates (time and space)
and Stokes and anti Stokes measurements. We can access the data by key.
It is presented as a DataArray. More examples are found at
http://xarray.pydata.org/en/stable/indexing.html

.. code:: ipython3

    ds['st']  # is the data stored, presented as a DataArray




.. raw:: html

    <div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
    <defs>
    <symbol id="icon-database" viewBox="0 0 32 32">
    <title>Show/Hide data repr</title>
    <path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
    <path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    <path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
    </symbol>
    <symbol id="icon-file-text2" viewBox="0 0 32 32">
    <title>Show/Hide attributes</title>
    <path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
    <path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    <path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
    </symbol>
    </defs>
    </svg>
    <style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
     *
     */
    
    :root {
      --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
      --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
      --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
      --xr-border-color: var(--jp-border-color2, #e0e0e0);
      --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
      --xr-background-color: var(--jp-layout-color0, white);
      --xr-background-color-row-even: var(--jp-layout-color1, white);
      --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
    }
    
    .xr-wrap {
      min-width: 300px;
      max-width: 700px;
    }
    
    .xr-header {
      padding-top: 6px;
      padding-bottom: 6px;
      margin-bottom: 4px;
      border-bottom: solid 1px var(--xr-border-color);
    }
    
    .xr-header > div,
    .xr-header > ul {
      display: inline;
      margin-top: 0;
      margin-bottom: 0;
    }
    
    .xr-obj-type,
    .xr-array-name {
      margin-left: 2px;
      margin-right: 10px;
    }
    
    .xr-obj-type {
      color: var(--xr-font-color2);
    }
    
    .xr-sections {
      padding-left: 0 !important;
      display: grid;
      grid-template-columns: 150px auto auto 1fr 20px 20px;
    }
    
    .xr-section-item {
      display: contents;
    }
    
    .xr-section-item input {
      display: none;
    }
    
    .xr-section-item input + label {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-item input:enabled + label {
      cursor: pointer;
      color: var(--xr-font-color2);
    }
    
    .xr-section-item input:enabled + label:hover {
      color: var(--xr-font-color0);
    }
    
    .xr-section-summary {
      grid-column: 1;
      color: var(--xr-font-color2);
      font-weight: 500;
    }
    
    .xr-section-summary > span {
      display: inline-block;
      padding-left: 0.5em;
    }
    
    .xr-section-summary-in:disabled + label {
      color: var(--xr-font-color2);
    }
    
    .xr-section-summary-in + label:before {
      display: inline-block;
      content: '►';
      font-size: 11px;
      width: 15px;
      text-align: center;
    }
    
    .xr-section-summary-in:disabled + label:before {
      color: var(--xr-disabled-color);
    }
    
    .xr-section-summary-in:checked + label:before {
      content: '▼';
    }
    
    .xr-section-summary-in:checked + label > span {
      display: none;
    }
    
    .xr-section-summary,
    .xr-section-inline-details {
      padding-top: 4px;
      padding-bottom: 4px;
    }
    
    .xr-section-inline-details {
      grid-column: 2 / -1;
    }
    
    .xr-section-details {
      display: none;
      grid-column: 1 / -1;
      margin-bottom: 5px;
    }
    
    .xr-section-summary-in:checked ~ .xr-section-details {
      display: contents;
    }
    
    .xr-array-wrap {
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: 20px auto;
    }
    
    .xr-array-wrap > label {
      grid-column: 1;
      vertical-align: top;
    }
    
    .xr-preview {
      color: var(--xr-font-color3);
    }
    
    .xr-array-preview,
    .xr-array-data {
      padding: 0 5px !important;
      grid-column: 2;
    }
    
    .xr-array-data,
    .xr-array-in:checked ~ .xr-array-preview {
      display: none;
    }
    
    .xr-array-in:checked ~ .xr-array-data,
    .xr-array-preview {
      display: inline-block;
    }
    
    .xr-dim-list {
      display: inline-block !important;
      list-style: none;
      padding: 0 !important;
      margin: 0;
    }
    
    .xr-dim-list li {
      display: inline-block;
      padding: 0;
      margin: 0;
    }
    
    .xr-dim-list:before {
      content: '(';
    }
    
    .xr-dim-list:after {
      content: ')';
    }
    
    .xr-dim-list li:not(:last-child):after {
      content: ',';
      padding-right: 5px;
    }
    
    .xr-has-index {
      font-weight: bold;
    }
    
    .xr-var-list,
    .xr-var-item {
      display: contents;
    }
    
    .xr-var-item > div,
    .xr-var-item label,
    .xr-var-item > .xr-var-name span {
      background-color: var(--xr-background-color-row-even);
      margin-bottom: 0;
    }
    
    .xr-var-item > .xr-var-name:hover span {
      padding-right: 5px;
    }
    
    .xr-var-list > li:nth-child(odd) > div,
    .xr-var-list > li:nth-child(odd) > label,
    .xr-var-list > li:nth-child(odd) > .xr-var-name span {
      background-color: var(--xr-background-color-row-odd);
    }
    
    .xr-var-name {
      grid-column: 1;
    }
    
    .xr-var-dims {
      grid-column: 2;
    }
    
    .xr-var-dtype {
      grid-column: 3;
      text-align: right;
      color: var(--xr-font-color2);
    }
    
    .xr-var-preview {
      grid-column: 4;
    }
    
    .xr-var-name,
    .xr-var-dims,
    .xr-var-dtype,
    .xr-preview,
    .xr-attrs dt {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding-right: 10px;
    }
    
    .xr-var-name:hover,
    .xr-var-dims:hover,
    .xr-var-dtype:hover,
    .xr-attrs dt:hover {
      overflow: visible;
      width: auto;
      z-index: 1;
    }
    
    .xr-var-attrs,
    .xr-var-data {
      display: none;
      background-color: var(--xr-background-color) !important;
      padding-bottom: 5px !important;
    }
    
    .xr-var-attrs-in:checked ~ .xr-var-attrs,
    .xr-var-data-in:checked ~ .xr-var-data {
      display: block;
    }
    
    .xr-var-data > table {
      float: right;
    }
    
    .xr-var-name span,
    .xr-var-data,
    .xr-attrs {
      padding-left: 25px !important;
    }
    
    .xr-attrs,
    .xr-var-attrs,
    .xr-var-data {
      grid-column: 1 / -1;
    }
    
    dl.xr-attrs {
      padding: 0;
      margin: 0;
      display: grid;
      grid-template-columns: 125px auto;
    }
    
    .xr-attrs dt, dd {
      padding: 0;
      margin: 0;
      float: left;
      padding-right: 10px;
      width: auto;
    }
    
    .xr-attrs dt {
      font-weight: normal;
      grid-column: 1;
    }
    
    .xr-attrs dt:hover span {
      display: inline-block;
      background: var(--xr-background-color);
      padding-right: 10px;
    }
    
    .xr-attrs dd {
      grid-column: 2;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .xr-icon-database,
    .xr-icon-file-text2 {
      display: inline-block;
      vertical-align: middle;
      width: 1em;
      height: 1.5em !important;
      stroke-width: 0;
      stroke: currentColor;
      fill: currentColor;
    }
    </style><div class='xr-wrap'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'st'</div><ul class='xr-dim-list'><li><span class='xr-has-index'>x</span>: 1461</li><li><span class='xr-has-index'>time</span>: 3</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-52baf23e-a792-47af-ac5c-af9cb0875295' class='xr-array-in' type='checkbox' ><label for='section-52baf23e-a792-47af-ac5c-af9cb0875295' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>-0.8058 0.4287 -0.513 -0.4589 -0.1245 ... 38.32 27.99 27.83 28.81</span></div><pre class='xr-array-data'>array([[-8.05791e-01,  4.28741e-01, -5.13021e-01],
           [-4.58870e-01, -1.24484e-01,  9.68469e-03],
           [ 4.89174e-01, -9.57734e-02,  5.62837e-02],
           ...,
           [ 4.68457e+01,  4.72201e+01,  4.79139e+01],
           [ 3.76634e+01,  3.74649e+01,  3.83160e+01],
           [ 2.79879e+01,  2.78331e+01,  2.88055e+01]])</pre></div></li><li class='xr-section-item'><input id='section-c52543ee-3b72-41f5-a609-5e927ed6a6b6' class='xr-section-summary-in' type='checkbox'  checked><label for='section-c52543ee-3b72-41f5-a609-5e927ed6a6b6' class='xr-section-summary' >Coordinates: <span>(7)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-80.74 -80.62 ... 104.7 104.8</div><input id='attrs-2d5e7da7-c0db-4757-8bd4-0f12feda8a80' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-2d5e7da7-c0db-4757-8bd4-0f12feda8a80' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-b91849f8-3462-4169-9eae-96252f42641d' class='xr-var-data-in' type='checkbox'><label for='data-b91849f8-3462-4169-9eae-96252f42641d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>distance</dd><dt><span>description :</span></dt><dd>Length along fiber</dd><dt><span>long_description :</span></dt><dd>Starting at connector of forward channel</dd><dt><span>units :</span></dt><dd>m</dd></dl></div><pre class='xr-var-data'>array([-80.7443, -80.6172, -80.4901, ..., 104.567 , 104.694 , 104.821 ])</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>filename</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>&lt;U31</div><div class='xr-var-preview xr-preview'>&#x27;channel 2_20180504132202074.xml&#x27; ... &#x27;channel 2_20180504132303723.xml&#x27;</div><input id='attrs-3e546c5d-b1b2-40a6-a998-22d72f97be27' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-3e546c5d-b1b2-40a6-a998-22d72f97be27' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3fe18043-ade0-43ce-b842-c926083ae1a0' class='xr-var-data-in' type='checkbox'><label for='data-3fe18043-ade0-43ce-b842-c926083ae1a0' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([&#x27;channel 2_20180504132202074.xml&#x27;,
           &#x27;channel 2_20180504132232903.xml&#x27;,
           &#x27;channel 2_20180504132303723.xml&#x27;], dtype=&#x27;&lt;U31&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>filename_tstamp</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>20180504132202074 ... 20180504132303723</div><input id='attrs-2e25c5b8-140d-43d8-9c80-c417829ffa73' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-2e25c5b8-140d-43d8-9c80-c417829ffa73' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-fc7b49ed-dee2-414b-b20a-0ffaeacd3897' class='xr-var-data-in' type='checkbox'><label for='data-fc7b49ed-dee2-414b-b20a-0ffaeacd3897' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([20180504132202074, 20180504132232903, 20180504132303723])</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>timestart</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2018-05-04T12:22:02.710000 ... 2018-05-04T12:23:03.716000</div><input id='attrs-2e3e9c03-5214-4688-a3ed-3c2c0dc989a1' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-2e3e9c03-5214-4688-a3ed-3c2c0dc989a1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6d0a899c-e44b-4efc-8e04-4c650eeeddc1' class='xr-var-data-in' type='checkbox'><label for='data-6d0a899c-e44b-4efc-8e04-4c650eeeddc1' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time start of the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><pre class='xr-var-data'>array([&#x27;2018-05-04T12:22:02.710000000&#x27;, &#x27;2018-05-04T12:22:32.702000000&#x27;,
           &#x27;2018-05-04T12:23:03.716000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>timeend</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2018-05-04T12:22:32.710000 ... 2018-05-04T12:23:33.716000</div><input id='attrs-8ccb0ac7-dc33-4309-97b8-fb7e3e0016f1' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-8ccb0ac7-dc33-4309-97b8-fb7e3e0016f1' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-9db7b365-b372-4f76-98b3-3a83abc86623' class='xr-var-data-in' type='checkbox'><label for='data-9db7b365-b372-4f76-98b3-3a83abc86623' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time end of the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><pre class='xr-var-data'>array([&#x27;2018-05-04T12:22:32.710000000&#x27;, &#x27;2018-05-04T12:23:02.702000000&#x27;,
           &#x27;2018-05-04T12:23:33.716000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>time</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>datetime64[ns]</div><div class='xr-var-preview xr-preview'>2018-05-04T12:22:17.710000 ... 2018-05-04T12:23:18.716000</div><input id='attrs-757858d9-a887-4701-aecf-7c1be2714270' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-757858d9-a887-4701-aecf-7c1be2714270' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-08317245-958d-4c78-b3dd-372de5d3cbb4' class='xr-var-data-in' type='checkbox'><label for='data-08317245-958d-4c78-b3dd-372de5d3cbb4' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>time halfway the measurement</dd><dt><span>timezone :</span></dt><dd>UTC</dd></dl></div><pre class='xr-var-data'>array([&#x27;2018-05-04T12:22:17.710000000&#x27;, &#x27;2018-05-04T12:22:47.702000000&#x27;,
           &#x27;2018-05-04T12:23:18.716000000&#x27;], dtype=&#x27;datetime64[ns]&#x27;)</pre></li><li class='xr-var-item'><div class='xr-var-name'><span>acquisitiontimeFW</span></div><div class='xr-var-dims'>(time)</div><div class='xr-var-dtype'>timedelta64[ns]</div><div class='xr-var-preview xr-preview'>00:00:30 00:00:30 00:00:30</div><input id='attrs-bbbf9423-3cd6-4acc-b036-31f454d095d9' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-bbbf9423-3cd6-4acc-b036-31f454d095d9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1d238a36-da65-4c21-bd27-7cdebeba761e' class='xr-var-data-in' type='checkbox'><label for='data-1d238a36-da65-4c21-bd27-7cdebeba761e' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>description :</span></dt><dd>Acquisition time of the forward measurement</dd></dl></div><pre class='xr-var-data'>array([30000000000, 30000000000, 30000000000], dtype=&#x27;timedelta64[ns]&#x27;)</pre></li></ul></div></li><li class='xr-section-item'><input id='section-df7421f0-b23d-4ed5-847a-111d357ac3b8' class='xr-section-summary-in' type='checkbox'  checked><label for='section-df7421f0-b23d-4ed5-847a-111d357ac3b8' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>name :</span></dt><dd>st</dd><dt><span>description :</span></dt><dd>Stokes intensity</dd><dt><span>units :</span></dt><dd>-</dd></dl></div></li></ul></div></div>



.. code:: ipython3

    ds['tmp'].plot(figsize=(12, 8));

1 mean, min, max
----------------

The first argument is the dimension. The function is taken along that
dimension. ``dim`` can be any dimension (e.g., ``time``, ``x``). The
returned ``DataStore`` does not contain that dimension anymore.

Normally, you would like to keep the attributes (the informative texts
from the loaded files), so set ``keep_attrs`` to ``True``. They don’t
take any space compared to your Stokes data, so keep them.

Note that also the sections are stored as attribute. If you delete the
attributes, you would have to redefine the sections.

.. code:: ipython3

    ds_min = ds.mean(dim='time', keep_attrs=True)  # take the minimum of all data variables (e.g., Stokes, Temperature) along the time dimension

.. code:: ipython3

    ds_max = ds.max(dim='x', keep_attrs=True)  # Take the maximum of all data variables (e.g., Stokes, Temperature) along the x dimension

.. code:: ipython3

    ds_std = ds.std(dim='time', keep_attrs=True)  # Calculate the standard deviation along the time dimension

2 Selecting
-----------

What if you would like to get the maximum temperature between
:math:`x >= 20` m and :math:`x < 35` m over time? We first have to
select a section along the cable.

.. code:: ipython3

    section = slice(20., 35.)
    section_of_interest = ds.sel(x=section)

.. code:: ipython3

    section_of_interest_max = section_of_interest.max(dim='x')

What if you would like to have the measurement at approximately
:math:`x=20` m?

.. code:: ipython3

    point_of_interest = ds.sel(x=20., method='nearest')

3 Selecting by index
--------------------

What if you would like to see what the values on the first timestep are?
We can use isel (index select)

.. code:: ipython3

    section_of_interest = ds.isel(time=slice(0, 2))  # The first two time steps

.. code:: ipython3

    section_of_interest = ds.isel(x=0)

4 Downsample (time dimension)
-----------------------------

We currently have measurements at 3 time steps, with 30.001 seconds
inbetween. For our next exercise we would like to down sample the
measurements to 2 time steps with 47 seconds inbetween. The calculated
variances are not valid anymore. We use the function
``resample_datastore``.

.. code:: ipython3

    ds_resampled = ds.resample_datastore(how='mean', time="47S")

5 Upsample / Interpolation (length and time dimension)
------------------------------------------------------

So we have measurements every 0.12 cm starting at :math:`x=0` m. What if
we would like to change our coordinate system to have a value every 12
cm starting at :math:`x=0.05` m. We use (linear) interpolation,
extrapolation is not supported. The calculated variances are not valid
anymore.

.. code:: ipython3

    x_old = ds.x.data
    x_new = x_old[:-1] + 0.05 # no extrapolation
    ds_xinterped = ds.interp(coords={'x': x_new})

We can do the same in the time dimension

.. code:: ipython3

    import numpy as np
    time_old = ds.time.data
    time_new = time_old + np.timedelta64(10, 's')
    ds_tinterped = ds.interp(coords={'time': time_new})

