import React, { Component } from 'react';
import './App.css';
import './assets/react-toolbox/theme.css';
import Slider from 'react-toolbox/lib/slider/Slider';
import Tab from 'react-toolbox/lib/tabs/Tab';
import Tabs from 'react-toolbox/lib/tabs/Tabs';

import theme from './assets/react-toolbox/theme'
import ThemeProvider from 'react-toolbox/lib/ThemeProvider';

class App extends Component {
  constructor(props) {
    super(props)
    this.state = {
      step: 100002,
      batch: 1,
      fixedIndex: '0',
    };

    this.batchSize = 50;
    this.stepMap = {
      '0': 100002,
      '1': 90030,
      '2': 78375,
    }

    this.handleChange = this.handleChange.bind(this);
    this.handleFixedTabChange = this.handleFixedTabChange.bind(this);
  }

  handleChange(value) {
    this.setState({
      batch: value
    });
  }

  handleFixedTabChange(index) {
    this.setState({
      step: this.stepMap[index],
      fixedIndex: index
    });
  }

  render() {
    return (
      <ThemeProvider theme={theme}>
        <section>
          <Tabs index={this.state.fixedIndex} onChange={this.handleFixedTabChange} fixed>
            <Tab label='Naive ChangeGAN'><small></small></Tab>
            <Tab label='ChangeGAN with bbox and color loss'><small></small></Tab>
            <Tab label='ChangeGAN without bbox'><small></small></Tab>
          </Tabs>
          <div className="slider">
            <Slider pinned snaps min={1} max={10} step={1} editable value={this.state.batch} onChange={this.handleChange} />  
          </div>
            
          {[...Array(this.batchSize).keys()].map(idx => 
              <div className="image-section">
                <img src={`https://storage.googleapis.com/mlcampjeju2017-mlengine/eval-output/${this.state.step}/inputs_a_${this.state.batch}_${idx}.jpg`} alt=""/>
                <img src={`https://storage.googleapis.com/mlcampjeju2017-mlengine/eval-output/${this.state.step}/inputs_b_${this.state.batch}_${idx}.jpg`} alt=""/>
                <img src={`https://storage.googleapis.com/mlcampjeju2017-mlengine/eval-output/${this.state.step}/outputs_ba_${this.state.batch}_${idx}.jpg`} alt=""/>
                <img src={`https://storage.googleapis.com/mlcampjeju2017-mlengine/eval-output/${this.state.step}/outputs_ab_${this.state.batch}_${idx}.jpg`} alt=""/>
              </div>
            )}
        </section>
      </ThemeProvider>
    );
  }
}

export default App;
