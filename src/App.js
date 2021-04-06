import logo from './logo.svg';
import Title from './Title';
import { Button } from '@material-ui/core';
import React, { useState } from "react";
import { Grid } from '@material-ui/core';

import { makeStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import IconButton from "@material-ui/core/IconButton";
import MenuIcon from "@material-ui/icons/Menu";
import green from "@material-ui/core/colors/green";
import 'fontsource-roboto';


import HomeBar from './HomeBar'
import Content from './Content'
import Trump from './Trump.png'; // with import
import Biden from './Biden.png'; // with import



const App = () => {
    document.body.style.backgroundColor = "Black";
    return (
    <div className="App"    >
        <Grid container direction="column">
            <Grid item><HomeBar /></Grid>
            <Grid item container spacing={4}>
                <Grid item alignItems="flex-start">
                    <img src={Biden} />
                </Grid>
                <Grid item xs={5}>
                    <Content />
                </Grid>
                <Grid item alignItems="flex-end">
                    <img src={Trump} />
                </Grid>
            </Grid>
        </Grid>

        <div align="center">>
            <Button variant="contained" color="secondary">
                Analysis
            </Button>
            <Typography>YOYOYO</Typography>
        </div>

    </div>


  );
};

export default App;
