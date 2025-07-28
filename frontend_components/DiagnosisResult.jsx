
import React from 'react';
import { Card, List, Tag, Progress, Alert, Collapse } from 'antd';
import { InfoCircleOutlined, EyeOutlined } from '@ant-design/icons';

const { Panel } = Collapse;

const DiagnosisResult = ({ result }) => {
  if (!result) return null;

  const { category, confidence, message, details, possible_categories, confidence_level } = result;

  // 신뢰도에 따른 색상
  const getConfidenceColor = (level) => {
    switch (level) {
      case '높음': return 'green';
      case '중간': return 'orange';
      case '낮음': return 'red';
      default: return 'blue';
    }
  };

  return (
    <div className="diagnosis-result">
      {/* 정상인 경우 */}
      {category === '정상' && (
        <Alert
          message="진단 결과: 정상"
          description={message}
          type="success"
          showIcon
          icon={<EyeOutlined />}
        />
      )}

      {/* 불확실한 경우 */}
      {category === '불확실' && (
        <>
          <Alert
            message="명확한 진단 불가"
            description={message}
            type="warning"
            showIcon
          />
          <div style={{ marginTop: 16 }}>
            <h4>가능한 질환 분류:</h4>
            {possible_categories.map((cat, index) => (
              <Card key={index} style={{ marginBottom: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h4>{cat.name}</h4>
                  <span>{(cat.probability * 100).toFixed(1)}%</span>
                </div>
                <p>{cat.details.description}</p>
              </Card>
            ))}
          </div>
        </>
      )}

      {/* 확실한 진단 */}
      {category !== '정상' && category !== '불확실' && details && (
        <>
          <Card
            title={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>진단 결과: {category}</span>
                <Tag color={getConfidenceColor(confidence_level)}>
                  신뢰도: {confidence_level} ({(confidence * 100).toFixed(1)}%)
                </Tag>
              </div>
            }
          >
            <p style={{ fontSize: 16, marginBottom: 16 }}>{details.description}</p>
            
            <Collapse defaultActiveKey={['1']}>
              <Panel header="포함되는 세부 질환" key="1">
                <List
                  dataSource={details.common_diseases}
                  renderItem={disease => (
                    <List.Item>
                      <InfoCircleOutlined style={{ marginRight: 8 }} />
                      {disease}
                    </List.Item>
                  )}
                />
              </Panel>
              
              <Panel header="주요 증상" key="2">
                <List
                  dataSource={details.symptoms}
                  renderItem={symptom => (
                    <List.Item>• {symptom}</List.Item>
                  )}
                />
              </Panel>
            </Collapse>
            
            <Alert
              message="권장사항"
              description={details.recommendation}
              type="info"
              showIcon
              style={{ marginTop: 16 }}
            />
          </Card>
        </>
      )}
    </div>
  );
};

export default DiagnosisResult;
